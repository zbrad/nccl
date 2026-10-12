#!/bin/bash
# tuned/release.sh <variant> [--publish] [--skip-build]
#
# preflight gates -> build -> package -> verify the tarball -> (only with --publish) publish.
# Without --publish it is a dry run: everything up to and including verification really runs,
# but nothing is tagged, pushed or published.
#
#   CUDA_TAG=cu133 bash tuned/release.sh gb10               # dry run
#   CUDA_TAG=cu133 bash tuned/release.sh gb10 --publish     # for real
set -euo pipefail

REPODIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT=""
PUBLISH=0
SKIP_BUILD=0
for arg in "$@"; do
    case "${arg}" in
        --publish) PUBLISH=1 ;;
        --skip-build) SKIP_BUILD=1 ;;
        -*) echo "ERROR: unknown option ${arg}" >&2; exit 2 ;;
        *) VARIANT="${arg}" ;;
    esac
done
[[ -n "${VARIANT}" ]] || { echo "usage: tuned/release.sh <variant> [--publish] [--skip-build]" >&2; exit 2; }

# shellcheck source=env.sh
source "${REPODIR}/tuned/env.sh" "${VARIANT}" || exit 1
cd "${REPODIR}"

fail() { echo "FAIL: $*" >&2; exit 1; }
ok() { echo "ok:   $*"; }
BRANCH="tuned-builds"
DIST_DIR="${NCCL_TUNED_DIST_DIR}"

echo "===================================================="
echo "NCCL ${NCCL_TUNED_VERSION} ${GPU_TUNED_HW_LABEL} release ($([[ ${PUBLISH} -eq 1 ]] && echo PUBLISH || echo "dry run"))"
echo "===================================================="

# --- gates ---
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BRANCH}" ]] || fail "not on ${BRANCH}"
ok "on ${BRANCH}"
gpu_tuned_require_clean_tree "${REPODIR}" >/dev/null || fail "dirty tree (a build stamps its commit; commit first)"
ok "clean tree"
git remote get-url upstream >/dev/null 2>&1 || fail "no 'upstream' remote"
[[ -n "${GPU_TUNED_UPSTREAM_REF:-}" ]] && git rev-parse --verify -q "${GPU_TUNED_UPSTREAM_REF}^{commit}" >/dev/null \
    || fail "GPU_TUNED_UPSTREAM_REF='${GPU_TUNED_UPSTREAM_REF:-}' is not a ref in this clone"
ok "upstream ref ${GPU_TUNED_UPSTREAM_REF} resolves"
git merge-base --is-ancestor "${GPU_TUNED_UPSTREAM_REF}" HEAD || fail "HEAD does not contain ${GPU_TUNED_UPSTREAM_REF}"
ok "HEAD contains ${GPU_TUNED_UPSTREAM_REF}"
grep -q "zbrad/nccl tuned-builds: compile LDMC" src/device/Makefile || fail "single-arch fix marker missing from src/device/Makefile"
ok "single-arch fix present"
if [[ ${PUBLISH} -eq 1 && -n "${GPU_TUNED_LABEL_SEQ:-}" ]]; then
    fail "GPU_TUNED_LABEL_SEQ is set; publishing must read origin's tags for the sequence"
fi

# --- build + package ---
[[ ${SKIP_BUILD} -eq 1 ]] || bash tuned/build.sh "${VARIANT}"
NCCL_TUNED_NO_PUBLISH=1 bash tuned/package.sh "${VARIANT}"

# --- verify the tarball that would be published ---
TAG="$(cat "${DIST_DIR}/release-tag.txt")"
TARBALL="$(ls -t "${DIST_DIR}"/nccl-tuned-*.tar.gz | head -1)"
(cd "${DIST_DIR}" && sha256sum -c "$(basename "${TARBALL}").sha256" >/dev/null) || fail "checksum sidecar does not match ${TARBALL}"
ok "checksum sidecar matches"
CHECK="$(mktemp -d)"
trap 'rm -rf "${CHECK}"' EXIT
tar -xzf "${TARBALL}" -C "${CHECK}"
ROOT="$(find "${CHECK}" -mindepth 1 -maxdepth 1 -type d | head -1)"
for f in lib/libnccl_static.a include/nccl.h LICENSE.txt BUILD_INFO; do
    [[ -s "${ROOT}/${f}" ]] || fail "tarball is missing ${f}"
done
ok "tarball has lib, headers, license, BUILD_INFO"
export PATH="${CUDA_HOME}/bin:${PATH}"
gpu_tuned_verify_arch "${ROOT}/lib/libnccl_static.a" "${GPU_TUNED_CUDA_ARCH}" >/dev/null || fail "extracted static archive is not single-arch sm_${GPU_TUNED_CUDA_ARCH}"
ok "extracted archive is single-arch sm_${GPU_TUNED_CUDA_ARCH}"
grep -q "^tag=${TAG}$" "${ROOT}/BUILD_INFO" || fail "BUILD_INFO tag does not match ${TAG}"
ok "BUILD_INFO names ${TAG}"

if [[ ${PUBLISH} -eq 0 ]]; then
    echo ""
    echo "Dry run complete: ${TAG} verified, nothing published. Re-run with --publish."
    exit 0
fi

gpu_tuned_publish_release "zbrad/nccl" "${TAG}" "$(cat "${DIST_DIR}/release-title.txt")" "$(cat "${DIST_DIR}/release-notes.md")" \
    "${TARBALL}#$(basename "${TARBALL}")" \
    "${TARBALL}.sha256#$(basename "${TARBALL}").sha256"
echo "Done: https://github.com/zbrad/nccl/releases/tag/${TAG}"
