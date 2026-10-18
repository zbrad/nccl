#!/bin/bash
# tuned/package.sh <variant> — package an already built (tuned/build.sh) and tested (tuned/full_test.sh) tuned NCCL
# into a tarball: the static archive PyTorch links (USE_SYSTEM_NCCL=1 USE_STATIC_NCCL=1), the headers, the license,
# BUILD_INFO and BUILD_INFO.json. It does not ship libnccl.so: a shared library of ours would need its own unique
# SONAME next to NVIDIA's libnccl.so.2 (docs/ARCHITECTURE.md F1). This script NEVER tags, pushes or publishes:
# tuned/release.sh does that after its gates (docs/ARCHITECTURE.md B3). Stage order:
#   tuned/build.sh -> tuned/full_test.sh -> this -> tuned/smoke_test.sh -> (commit the release notes) -> tuned/release.sh
#
# Names follow tuned-common's native scheme. The date label is allocated ONCE, by tuned/build.sh (it is compiled into the
# library's version string, "NCCL version 2.32.3-gb10-cu134.20261018.0+cuda13.4"), recorded in <build root>/release-label,
# and read here, never recomputed (docs/ARCHITECTURE.md E4):
#   tag      v0.<YYYYMMDD>.<seq>-<variant>-cu<cuda>
#   tarball  nccl-tuned-<nccl version>-<variant>-cu<cuda>-<YYYYMMDD>[.<seq>].tar.gz
set -euo pipefail

REPODIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/package.sh <variant>}"
# shellcheck source=env.sh
source "${REPODIR}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}" || exit 1

BUILD_DIR="${NCCL_TUNED_BUILD_DIR}"
DIST_DIR="${NCCL_TUNED_DIST_DIR}"
RELEASES_DIR="${NCCL_TUNED_BUILD_ROOT}/releases"
TEST_LOG="${RELEASES_DIR}/TEST_RESULTS_${GPU_TUNED_VARIANT}.log"
STATIC="${BUILD_DIR}/lib/libnccl_static.a"
[[ -f "${STATIC}" && -f "${BUILD_DIR}/include/nccl.h" ]] || {
    echo "ERROR: ${BUILD_DIR} has no lib/libnccl_static.a and include/nccl.h. Run tuned/build.sh ${GPU_TUNED_VARIANT} first." >&2
    exit 1
}

# The header must say the version this tree declares: a stale build dir is the likely cause.
HDR_VER="$(awk '$2=="NCCL_MAJOR"{M=$3} $2=="NCCL_MINOR"{N=$3} $2=="NCCL_PATCH"{P=$3} END{printf "%s.%s.%s", M, N, P}' "${BUILD_DIR}/include/nccl.h")"
[[ "${HDR_VER}" == "${NCCL_TUNED_VERSION%%[!0-9.]*}" ]] || {
    echo "ERROR: ${BUILD_DIR}/include/nccl.h says NCCL ${HDR_VER}, this tree is ${NCCL_TUNED_VERSION}: stale build. Rebuild." >&2
    exit 1
}

# A package is made only from a build that passed the tests: the log must exist, show a pass, name this exact commit,
# and be newer than the archive (the build tree is named after the commit, so this is the same source).
[[ -f "${TEST_LOG}" ]] || { echo "ERROR: ${TEST_LOG} not found; run tuned/full_test.sh ${GPU_TUNED_VARIANT} for ${TOOLKIT_ID} before packaging." >&2; exit 1; }
grep -q '^=== Full test suite summary: passed ===$' "${TEST_LOG}" || { echo "ERROR: ${TEST_LOG} does not show a pass." >&2; exit 1; }
[[ "$(sed -n 's/^commit: *//p' "${TEST_LOG}" | head -1)" == "$(git -C "${REPODIR}" rev-parse HEAD)" ]] || {
    echo "ERROR: ${TEST_LOG} was run at another commit than HEAD; re-run tuned/full_test.sh." >&2; exit 1; }
[[ ! "${STATIC}" -nt "${TEST_LOG}" ]] || { echo "ERROR: ${STATIC} is newer than the test log; re-run tuned/full_test.sh." >&2; exit 1; }

export PATH="${CUDA_HOME}/bin:${PATH}"
gpu_tuned_verify_arch "${STATIC}" "${GPU_TUNED_CUDA_ARCH}" || exit 1

rm -f "${DIST_DIR}/package-info.txt"
# The label was compiled into this build; the headers about to be packaged must say so.
[[ -n "${NCCL_TUNED_LABEL}" ]] || { echo "ERROR: ${NCCL_TUNED_LABEL_FILE} is missing: the build predates the label in the version string. Run tuned/build.sh ${GPU_TUNED_VARIANT}." >&2; exit 1; }
grep -q "^#define NCCL_SUFFIX \"${NCCL_TUNED_SUFFIX}\"" "${BUILD_DIR}/include/nccl.h" || {
    echo "ERROR: ${BUILD_DIR}/include/nccl.h does not define NCCL_SUFFIX \"${NCCL_TUNED_SUFFIX}\": stale build. Rebuild." >&2; exit 1; }
LABEL="${NCCL_TUNED_LABEL}"
RELEASE_TAG="$(gpu_tuned_native_tag "${GPU_TUNED_VARIANT}" "${TOOLKIT_ID}" "${LABEL}")" || exit 1
NAME="nccl-tuned-${NCCL_TUNED_VERSION}-${GPU_TUNED_VARIANT}-${TOOLKIT_ID}-${LABEL}"

mkdir -p "${DIST_DIR}"
STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT
mkdir -p "${STAGE}/${NAME}/lib"
cp -p "${STATIC}" "${STAGE}/${NAME}/lib/"
cp -a "${BUILD_DIR}/include" "${STAGE}/${NAME}/include"
cp -p "${REPODIR}/LICENSE.txt" "${STAGE}/${NAME}/"
{
    echo "name=${NAME}"
    echo "tag=${RELEASE_TAG}"
    echo "nccl_version=${NCCL_TUNED_VERSION}"
    echo "nccl_version_string=${NCCL_TUNED_VERSION}${NCCL_TUNED_SUFFIX}"
    echo "source=${NCCL_TUNED_DESCRIBE}"
    echo "upstream_tag=${GPU_TUNED_UPSTREAM_REF}"
    echo "variant=${GPU_TUNED_VARIANT}"
    echo "arch=sm_${GPU_TUNED_CUDA_ARCH}"
    echo "toolkit=${TOOLKIT_ID} (CUDA ${TOOLKIT_CUDA_VER})"
    echo "provenance=$(gpu_tuned_git_provenance "${REPODIR}")"
    echo "static_sha256=$(sha256sum "${STAGE}/${NAME}/lib/libnccl_static.a" | cut -d' ' -f1)"
} > "${STAGE}/${NAME}/BUILD_INFO"
# The same identity as JSON (docs/ARCHITECTURE.md H7). A static archive cannot carry an ELF section, so this file is
# the carrier. release first; deps lists what the archive was built from.
NVCC_FULL="$("${CUDA_HOME}/bin/nvcc" --version | sed -n 's/.*release [0-9.]*, V\([0-9.]*\).*/\1/p' | head -1)"
( cd "${REPODIR}" && gpu_tuned_write_build_info_json "${STAGE}/${NAME}/BUILD_INFO.json" \
    package=nccl component=libnccl_static.a "release=${RELEASE_TAG}" "variant=${GPU_TUNED_VARIANT}" \
    "hardware=${GPU_TUNED_HW_LABEL}" "arch=sm_${GPU_TUNED_CUDA_ARCH}" \
    "cuda.tag=${TOOLKIT_ID}" "cuda.version=${TOOLKIT_CUDA_VER}" "cuda.nvcc=${NVCC_FULL:-unknown}" \
    "nccl.version=${NCCL_TUNED_VERSION}${NCCL_TUNED_SUFFIX}" "upstream.declared_version=${NCCL_TUNED_VERSION}" \
    "deps.static_sha256=$(sha256sum "${STAGE}/${NAME}/lib/libnccl_static.a" | cut -d' ' -f1)" ) || exit 1

TARBALL="${DIST_DIR}/${NAME}.tar.gz"
tar -C "${STAGE}" -czf "${TARBALL}" "${NAME}"
echo "Tarball: ${TARBALL} ($(du -sh "${TARBALL}" | awk '{print $1}'))"
gpu_tuned_write_sha256_sidecar "${TARBALL}" || exit 1

# Release notes: a tracked, reviewed file (tuned/releases/RELEASE_NOTES_<tag without v>.md). When it does not exist yet, a DRAFT
# is written here for review; release.sh refuses a file that still carries the DRAFT banner. Numbers come from the test log.
NOTES_FILE="${REPODIR}/tuned/releases/RELEASE_NOTES_${RELEASE_TAG#v}.md"
if [[ ! -f "${NOTES_FILE}" ]]; then
    mkdir -p "$(dirname "${NOTES_FILE}")"
    STATIC_CHECKS="$(sed -n 's/^  ok:   static: PASSED \([0-9]*\) checks$/\1/p' "${TEST_LOG}" | head -1)"
    SHARED_CHECKS="$(sed -n 's/^  ok:   shared: PASSED \([0-9]*\) checks$/\1/p' "${TEST_LOG}" | head -1)"
    cat > "${NOTES_FILE}" <<NOTES
**DRAFT — not yet published.** Review, correct and remove this line, then commit the file.

# NCCL ${NCCL_TUNED_VERSION} for ${GPU_TUNED_HW_LABEL} (${TOOLKIT_ID}) — ${RELEASE_TAG}

Static archive and headers of NCCL ${NCCL_TUNED_VERSION}, built single-arch (sm_${GPU_TUNED_CUDA_ARCH}) with CUDA ${TOOLKIT_CUDA_VER}, for
PyTorch's USE_SYSTEM_NCCL=1 USE_STATIC_NCCL=1 and for llama.cpp's static NCCL. No shared library is shipped.

## Version
The library reports itself as \`NCCL version ${NCCL_TUNED_VERSION}${NCCL_TUNED_SUFFIX}+cuda${TOOLKIT_CUDA_VER}\` (also in \`nccl.h\` as \`NCCL_SUFFIX "${NCCL_TUNED_SUFFIX}"\`):
NCCL ${NCCL_TUNED_VERSION}, built for variant \`${GPU_TUNED_VARIANT}\` with toolkit \`${TOOLKIT_ID}\`, release label \`${LABEL%%.*}.$(gpu_tuned_label_seq "${LABEL}")\` (the same label as the tag
\`${RELEASE_TAG}\`); \`+cuda${TOOLKIT_CUDA_VER}\` is NCCL's own record of the CUDA it was compiled with. The numeric version code and the
library file names are unchanged.

## Source
- Base: NVIDIA/nccl tag \`${GPU_TUNED_UPSTREAM_REF}\` (the NCCL release NVIDIA ships for CUDA ${TOOLKIT_CUDA_VER}), plus this fork's changes: the
  single-arch fix in src/device/Makefile (NVIDIA/nccl issue 2420, still open upstream) and the \`tuned/\` build tooling.
- Fork commit: \`$(git -C "${REPODIR}" rev-parse HEAD)\` (\`${NCCL_TUNED_DESCRIBE}\`).

## Tested
tuned/tests/nccl_single_rank.cu on one GB10 GPU: ${STATIC_CHECKS:-?} checks against the static archive and ${SHARED_CHECKS:-?} against the shared build,
all passed. **Single rank only:** no multi-rank algorithm or transport (NVLink, network, shared memory) was exercised, and neither
were the multi-rank device kernels. FP8 average is rejected with an invalid-argument error, as in NVIDIA's own 2.31.2 package.

## Files
\`nccl-tuned-…tar.gz\` with \`lib/libnccl_static.a\`, \`include/\`, \`LICENSE.txt\`, \`BUILD_INFO\`, \`BUILD_INFO.json\`; a \`.sha256\` sidecar; the test and smoke logs.
NOTES
    echo "Draft release notes written: ${NOTES_FILE#"${REPODIR}"/} (review, remove the DRAFT line, commit)"
fi

# Written last: its presence means the package passed every check above. release.sh and smoke_test.sh read it.
cat > "${DIST_DIR}/package-info.txt" <<INFO
release_tag=${RELEASE_TAG}
tarball=$(basename "${TARBALL}")
nccl_version=${NCCL_TUNED_VERSION}
nccl_version_string=${NCCL_TUNED_VERSION}${NCCL_TUNED_SUFFIX}
source=${NCCL_TUNED_DESCRIBE}
upstream_tag=${GPU_TUNED_UPSTREAM_REF}
commit=$(git -C "${REPODIR}" rev-parse HEAD)
cuda_tag=${TOOLKIT_ID}
variant=${GPU_TUNED_VARIANT}
sha256=$(sha256sum "${TARBALL}" | awk '{print $1}')
build_root=${NCCL_TUNED_BUILD_ROOT}
INFO
# release.sh runs after the notes commit, when HEAD (and so the <describe> tree name) has moved: it finds this package here.
printf '%s\n' "${NCCL_TUNED_BUILD_ROOT}" > "${REPODIR}/build/${TOOLKIT_ID}/last-package-root.txt"

echo ""
echo "Packaged (not published): ${TARBALL} (${RELEASE_TAG})"
echo "Next: bash tuned/smoke_test.sh ${GPU_TUNED_VARIANT}, commit the release notes, then bash tuned/release.sh ${GPU_TUNED_VARIANT}"
