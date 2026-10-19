#!/bin/bash
# tuned/release.sh <variant> — the release gates for the tarball tuned/package.sh made. It never publishes.
# Stage order, as in raft, cuvs, faiss and llama.cpp:
#   tuned/build.sh -> tuned/full_test.sh -> tuned/package.sh -> tuned/smoke_test.sh
#   -> (review and commit tuned/releases/RELEASE_NOTES_<label>-<variant>-<cuda>.md) -> this -> tuned/publish.sh.
# It never builds, tests or packages. It runs the shared gates (gpu_tuned_release_gates) plus nccl's own, and only if all
# pass writes a ready-marker (gpu_tuned_release_ready) that tuned/publish.sh publishes from. A failing run removes any
# old marker first, so it cannot leave one behind. docs/ARCHITECTURE.md (tuned-common) lists the gates (I1, I5).
#
# The release is named by the date label package.sh allocated (v0.<YYYYMMDD>.<seq>-<variant>-<cuda>), read from package-info.txt
# and never recomputed here: the notes commit moves HEAD and its date. Toolkit branches: a cu133 release is made from
# tuned-builds-cu133 and a cu134 one from tuned-builds-cu134 (each is based on the NCCL release NVIDIA ships for that CUDA).
#
#   CUDA_TAG=cu134 bash tuned/release.sh gb10     # gates only; on success writes the ready-marker
#   CUDA_TAG=cu134 bash tuned/publish.sh gb10    # publishes what the gates passed
set -uo pipefail

REPODIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="${1:-}"
case "${VARIANT}" in
    "") echo "usage: tuned/release.sh <variant>" >&2; exit 2 ;;
    -*) echo "ERROR: unknown option ${VARIANT}" >&2; echo "usage: tuned/release.sh <variant>" >&2; exit 2 ;;
esac
[[ $# -eq 1 ]] || { echo "ERROR: tuned/release.sh takes one argument, the variant" >&2; exit 2; }

# shellcheck source=env.sh
source "${REPODIR}/tuned/env.sh" "${VARIANT}" || exit 1
cd "${REPODIR}"

DIST_DIR="${NCCL_TUNED_DIST_DIR}"
RELEASES_DIR="${NCCL_TUNED_BUILD_ROOT}/releases"
MARKER="${RELEASES_DIR}/RELEASE_READY_${VARIANT}"
INFO="${DIST_DIR}/package-info.txt"
TEST_LOG="${RELEASES_DIR}/TEST_RESULTS_${VARIANT}.log"
SMOKE_LOG="${RELEASES_DIR}/SMOKE_RESULTS_${VARIANT}.log"
BUILT="${NCCL_TUNED_BUILD_DIR}/lib/libnccl_static.a"
FAILURES=0
fail() { echo "  FAIL: $*" >&2; FAILURES=$((FAILURES + 1)); }
ok() { echo "  ok:   $*"; }

gpu_tuned_release_marker_clear "${MARKER}"

if [[ ! -f "${INFO}" ]]; then
    echo "ERROR: ${INFO} not found -- run 'bash tuned/package.sh ${VARIANT}' at the commit being released first." >&2
    exit 1
fi
info() { sed -n "s/^$1=//p" "${INFO}"; }
RELEASE_TAG="$(info release_tag)"; TARBALL="${DIST_DIR}/$(info tarball)"
PACKAGED_COMMIT="$(info commit)"; SOURCE="$(info source)"; NCCL_VER="$(info nccl_version)"
RELEASE_NOTES_FILE="tuned/releases/RELEASE_NOTES_${RELEASE_TAG#v}.md"
RELEASE_TITLE="NCCL ${NCCL_VER} (${SOURCE}) — ${GPU_TUNED_HW_LABEL} (${TOOLKIT_ID}), ${RELEASE_TAG#v}"

echo "===================================================="
echo "NCCL ${NCCL_VER} ${GPU_TUNED_HW_LABEL} release gates ${RELEASE_TAG}"
echo "===================================================="

# --- nccl-specific: source and branch ---
BRANCH="tuned-builds-${TOOLKIT_ID}"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BRANCH}" ]] && ok "on ${BRANCH}" || fail "not on ${BRANCH} (a ${TOOLKIT_ID} release is made from that branch)"
if [[ -n "${GPU_TUNED_UPSTREAM_REF:-}" ]] && git rev-parse --verify -q "${GPU_TUNED_UPSTREAM_REF}^{commit}" >/dev/null; then
    git merge-base --is-ancestor "${GPU_TUNED_UPSTREAM_REF}" HEAD && ok "HEAD contains upstream ${GPU_TUNED_UPSTREAM_REF}" || fail "HEAD does not contain ${GPU_TUNED_UPSTREAM_REF}"
else
    fail "GPU_TUNED_UPSTREAM_REF='${GPU_TUNED_UPSTREAM_REF:-}' is not a ref in this clone"
fi
grep -q "zbrad/nccl tuned-builds: compile LDMC" src/device/Makefile && ok "single-arch fix present" || fail "single-arch fix marker missing from src/device/Makefile"
grep -q "zbrad/nccl tuned-builds: version line" src/init.cc && ok "version-line patch present" || fail "version-line patch marker missing from src/init.cc"

# --- nccl-specific: the tag is free on origin (the shared gates check its shape and the clean, pushed tree) ---
if [[ -z "$(git ls-remote --tags origin "refs/tags/${RELEASE_TAG}" 2>/dev/null)" ]]; then ok "tag ${RELEASE_TAG} is not yet taken on origin"; else fail "tag ${RELEASE_TAG} already exists on origin: package again for the next sequence"; fi

# --- shared gates: tag shape, clean tree, pushed HEAD, test log fresh and passing, tarball and sha256 sidecar,
#     smoke log, package-info for this toolkit and variant, only release notes changed, notes committed and not a DRAFT ---
gpu_tuned_release_gates \
    "tag=${RELEASE_TAG}" "variant=${VARIANT}" "cuda=${TOOLKIT_ID}" "info=${INFO}" \
    "built=${BUILT}" "test_log=${TEST_LOG}" "test_pass=^=== Full test suite summary: passed ===\$" \
    "tarball=${TARBALL}" "smoke_log=${SMOKE_LOG}" "notes=${RELEASE_NOTES_FILE}" "packaged=${PACKAGED_COMMIT}" \
    || FAILURES=$((FAILURES + 1))

# --- nccl-specific: the test log was run on the packaged build ---
if [[ -f "${TEST_LOG}" ]]; then
    if [[ "$(sed -n 's/^built_commit: *//p' "${TEST_LOG}" | head -1)" == "$(info build_commit)" ]]; then
        ok "the test log was run on the packaged build $(info build_commit)"
    else
        fail "the test log was run on another build than the packaged one ($(info build_commit))"
    fi
fi

if [[ ${FAILURES} -gt 0 ]]; then
    echo "" >&2
    echo "ERROR: ${FAILURES} gate(s) failed; no ready-marker was written, nothing can be published." >&2
    exit 1
fi

echo ""
gpu_tuned_release_ready "${MARKER}" "zbrad/nccl" "${RELEASE_TAG}" "${RELEASE_TITLE}" "${RELEASE_NOTES_FILE}" \
    "${TARBALL}#$(basename "${TARBALL}")" \
    "${TARBALL}.sha256#SHA-256 checksum" \
    "${TEST_LOG}#Test results (${VARIANT}, single rank)" \
    "${SMOKE_LOG}#Package smoke test results (${VARIANT})" || exit 1
