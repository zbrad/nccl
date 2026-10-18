#!/bin/bash
# tuned/release.sh <variant> [--publish] — publish the tarball tuned/package.sh made as a GitHub release, after the gates.
# Stage order, as in raft, cuvs, faiss and llama.cpp:
#   tuned/build.sh -> tuned/full_test.sh -> tuned/package.sh -> tuned/smoke_test.sh
#   -> (review and commit tuned/releases/RELEASE_NOTES_<label>-<variant>-<cuda>.md) -> this.
# It never builds, tests or packages. Without --publish it only runs the gates and says what it would publish (a dry run:
# nothing is tagged, pushed or published). docs/ARCHITECTURE.md (tuned-common) lists the gates.
#
# The release is named by the date label package.sh allocated (v0.<YYYYMMDD>.<seq>-<variant>-<cuda>), read from package-info.txt
# and never recomputed here: the notes commit moves HEAD and its date. Toolkit branches: a cu133 release is made from
# tuned-builds-cu133 and a cu134 one from tuned-builds-cu134 (each is based on the NCCL release NVIDIA ships for that CUDA).
#
#   CUDA_TAG=cu134 bash tuned/release.sh gb10              # gates only
#   CUDA_TAG=cu134 bash tuned/release.sh gb10 --publish    # for real
set -uo pipefail

REPODIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT=""
PUBLISH=0
for arg in "$@"; do
    case "${arg}" in
        --publish) PUBLISH=1 ;;
        -*) echo "ERROR: unknown option ${arg}" >&2; exit 2 ;;
        *) VARIANT="${arg}" ;;
    esac
done
[[ -n "${VARIANT}" ]] || { echo "usage: tuned/release.sh <variant> [--publish]" >&2; exit 2; }

# The package's tree is named after the commit it was made at; the notes commit has moved HEAD since, so follow the
# pointer package.sh left for this toolkit (unless the caller set NCCL_TUNED_BUILD_ROOT).
_tk="${CUDA_TAG:-}"
[[ -n "${_tk}" || -z "${CUDA_VER:-}" ]] || _tk="cu${CUDA_VER//./}"
if [[ -n "${_tk}" && -z "${NCCL_TUNED_BUILD_ROOT:-}" && -f "${REPODIR}/build/${_tk}/last-package-root.txt" ]]; then
    NCCL_TUNED_BUILD_ROOT="$(cat "${REPODIR}/build/${_tk}/last-package-root.txt")"; export NCCL_TUNED_BUILD_ROOT
fi
unset _tk
# shellcheck source=env.sh
source "${REPODIR}/tuned/env.sh" "${VARIANT}" || exit 1
cd "${REPODIR}"

DIST_DIR="${NCCL_TUNED_DIST_DIR}"
RELEASES_DIR="${NCCL_TUNED_BUILD_ROOT}/releases"
INFO="${DIST_DIR}/package-info.txt"
TEST_LOG="${RELEASES_DIR}/TEST_RESULTS_${VARIANT}.log"
SMOKE_LOG="${RELEASES_DIR}/SMOKE_RESULTS_${VARIANT}.log"
BUILT="${NCCL_TUNED_BUILD_DIR}/lib/libnccl_static.a"
FAILURES=0
fail() { echo "  FAIL: $*" >&2; FAILURES=$((FAILURES + 1)); }
ok() { echo "  ok:   $*"; }

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
echo "NCCL ${NCCL_VER} ${GPU_TUNED_HW_LABEL} release ${RELEASE_TAG} ($([[ ${PUBLISH} -eq 1 ]] && echo PUBLISH || echo "gates only"))"
echo "===================================================="

# --- source and branch ---
BRANCH="tuned-builds-${TOOLKIT_ID}"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BRANCH}" ]] && ok "on ${BRANCH}" || fail "not on ${BRANCH} (a ${TOOLKIT_ID} release is made from that branch)"
gpu_tuned_require_clean_tree "${REPODIR}" >/dev/null && ok "clean tree" || fail "dirty tracked tree"
gpu_tuned_require_pushed "${REPODIR}" >/dev/null 2>&1 && ok "HEAD is pushed to origin" || fail "HEAD is not on origin: push it (a release tags the commit it names)"
if [[ -n "${GPU_TUNED_UPSTREAM_REF:-}" ]] && git rev-parse --verify -q "${GPU_TUNED_UPSTREAM_REF}^{commit}" >/dev/null; then
    git merge-base --is-ancestor "${GPU_TUNED_UPSTREAM_REF}" HEAD && ok "HEAD contains upstream ${GPU_TUNED_UPSTREAM_REF}" || fail "HEAD does not contain ${GPU_TUNED_UPSTREAM_REF}"
else
    fail "GPU_TUNED_UPSTREAM_REF='${GPU_TUNED_UPSTREAM_REF:-}' is not a ref in this clone"
fi
grep -q "zbrad/nccl tuned-builds: compile LDMC" src/device/Makefile && ok "single-arch fix present" || fail "single-arch fix marker missing from src/device/Makefile"

# --- identity ---
gpu_tuned_check_release_tag "${RELEASE_TAG}" 2>/dev/null && ok "tag carries a date label" || fail "tag ${RELEASE_TAG} carries no date label"
if [[ -z "$(git ls-remote --tags origin "refs/tags/${RELEASE_TAG}" 2>/dev/null)" ]]; then ok "tag ${RELEASE_TAG} is not yet taken on origin"; else fail "tag ${RELEASE_TAG} already exists on origin: package again for the next sequence"; fi
{ grep -qxF "cuda_tag=${TOOLKIT_ID}" "${INFO}" && grep -qxF "variant=${VARIANT}" "${INFO}"; } && ok "package-info is for ${TOOLKIT_ID} / ${VARIANT}" || fail "package-info.txt is for another toolkit or variant"

# --- tests: a fresh pass at the packaged commit ---
if [[ ! -f "${BUILT}" ]]; then
    fail "${BUILT} not found -- cannot tell whether the test results are fresh; run tuned/build.sh ${VARIANT}"
elif [[ ! -f "${TEST_LOG}" ]]; then
    fail "${TEST_LOG} not found -- run tuned/full_test.sh ${VARIANT}"
elif [[ "${BUILT}" -nt "${TEST_LOG}" ]]; then
    fail "${BUILT} is newer than the test log: the results predate the build; re-run tuned/full_test.sh"
elif ! grep -q '^=== Full test suite summary: passed ===$' "${TEST_LOG}"; then
    fail "${TEST_LOG} does not show a pass"
elif [[ "$(sed -n 's/^commit: *//p' "${TEST_LOG}" | head -1)" != "${PACKAGED_COMMIT}" ]]; then
    fail "the test log was run at another commit than the packaged ${PACKAGED_COMMIT}"
else
    ok "full test suite passed at the packaged commit, log newer than the built archive"
fi

# --- package: checksum, order, smoke test ---
if [[ ! -f "${TARBALL}" ]]; then
    fail "${TARBALL} not found"
else
    [[ "$(sha256sum "${TARBALL}" | awk '{print $1}')" == "$(info sha256)" ]] && ok "tarball sha256 matches package-info.txt" || fail "tarball sha256 does not match package-info.txt"
    SHA_FILE="${TARBALL}.sha256"
    gpu_tuned_verify_sha256_sidecar "${TARBALL}" 2>/dev/null && ok "sha256 sidecar matches the tarball" || fail "sha256 sidecar missing or not matching the tarball"
    if [[ -f "${TEST_LOG}" && "${TEST_LOG}" -nt "${TARBALL}" ]]; then fail "the test log is newer than the tarball; package again after testing"; else ok "tarball was made after the tests"; fi
    if [[ ! -f "${SMOKE_LOG}" ]]; then
        fail "${SMOKE_LOG} not found -- run tuned/smoke_test.sh ${VARIANT}"
    elif [[ "${TARBALL}" -nt "${SMOKE_LOG}" ]]; then
        fail "${TARBALL} is newer than the smoke test log; re-run tuned/smoke_test.sh ${VARIANT}"
    elif ! grep -q '^=== Smoke test summary: passed ===$' "${SMOKE_LOG}"; then
        fail "${SMOKE_LOG} does not show a pass"
    else
        ok "smoke test of this tarball passed"
    fi
fi

# --- only the release notes may differ from the packaged commit; the notes are tracked and reviewed ---
changed="$(git diff --name-only "${PACKAGED_COMMIT}" HEAD -- . ':!tuned/releases' 2>/dev/null || echo '?')"
if [[ -z "${changed}" ]]; then ok "only release notes changed since the packaged commit ${PACKAGED_COMMIT:0:9}"; else fail "files other than tuned/releases/ changed since the packaged commit: $(echo "${changed}" | head -3 | tr '\n' ' ')"; fi
if [[ ! -f "${RELEASE_NOTES_FILE}" ]]; then
    fail "${RELEASE_NOTES_FILE} not found (package.sh drafts it: review and commit it)"
elif grep -q 'DRAFT — not yet published' "${RELEASE_NOTES_FILE}"; then
    fail "${RELEASE_NOTES_FILE} is still a draft (DRAFT banner present)"
elif ! git ls-files --error-unmatch "${RELEASE_NOTES_FILE}" >/dev/null 2>&1; then
    fail "${RELEASE_NOTES_FILE} is not committed"
else
    ok "release notes ${RELEASE_NOTES_FILE##*/} committed"
fi

if [[ ${FAILURES} -gt 0 ]]; then
    echo "" >&2
    echo "ERROR: ${FAILURES} gate(s) failed; nothing was published." >&2
    exit 1
fi
if [[ ${PUBLISH} -eq 0 ]]; then
    echo ""
    echo "Gates passed. Would publish ${RELEASE_TAG} ('${RELEASE_TITLE}') with the tarball, its checksum, and the test and smoke logs."
    echo "Nothing was tagged or published. Re-run with --publish."
    exit 0
fi

echo ""
echo "Publishing zbrad/nccl@${RELEASE_TAG}..."
gpu_tuned_publish_release "zbrad/nccl" "${RELEASE_TAG}" "${RELEASE_TITLE}" "@${RELEASE_NOTES_FILE}" \
    "${TARBALL}#$(basename "${TARBALL}")" \
    "${SHA_FILE}#SHA-256 checksum" \
    "${TEST_LOG}#Test results (${VARIANT}, single rank)" \
    "${SMOKE_LOG}#Package smoke test results (${VARIANT})" || exit 1
echo "Release: https://github.com/zbrad/nccl/releases/tag/${RELEASE_TAG}"
