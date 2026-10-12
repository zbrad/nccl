#!/bin/bash
# tuned/package.sh <variant> — package an already-built tuned NCCL (tuned/build.sh) into a
# tarball: the static archive PyTorch links (USE_SYSTEM_NCCL=1 USE_STATIC_NCCL=1), the headers,
# the license and a BUILD_INFO file. It does not ship libnccl.so: a shared library of ours would
# need its own unique SONAME next to NVIDIA's libnccl.so.2 (docs/VERSIONING.md, tuned-common).
#
# Names follow tuned-common's native scheme (date label from HEAD's UTC commit date plus the
# lowest unused sequence among origin's tags; GPU_TUNED_LABEL_SEQ=0 skips the lookup):
#   tag      v0.<YYYYMMDD>.<seq>-<variant>-cu<cuda>
#   tarball  nccl-tuned-<nccl version>-<variant>-cu<cuda>-<YYYYMMDD>[.<seq>].tar.gz
#
# Publishes only when run without NCCL_TUNED_NO_PUBLISH; tuned/release.sh sets it and publishes
# after verifying the tarball. Run directly, it publishes — prefer release.sh.
set -euo pipefail

REPODIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/package.sh <variant>}"
# shellcheck source=env.sh
source "${REPODIR}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}" || exit 1

BUILD_DIR="${NCCL_TUNED_BUILD_DIR}"
DIST_DIR="${NCCL_TUNED_DIST_DIR}"
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

export PATH="${CUDA_HOME}/bin:${PATH}"
gpu_tuned_verify_arch "${STATIC}" "${GPU_TUNED_CUDA_ARCH}"

LABEL="$(gpu_tuned_allocate_date_label "${REPODIR}")" || exit 1
RELEASE_TAG="$(gpu_tuned_native_tag "${GPU_TUNED_VARIANT}" "${TOOLKIT_ID}" "${LABEL}")" || exit 1
NAME="nccl-tuned-${NCCL_TUNED_VERSION}-${GPU_TUNED_VARIANT}-${TOOLKIT_ID}-${LABEL}"
RELEASE_TITLE="NCCL ${NCCL_TUNED_VERSION} (${NCCL_TUNED_DESCRIBE}) — ${GPU_TUNED_HW_LABEL} (${TOOLKIT_ID}), ${LABEL}"

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
    echo "source=${NCCL_TUNED_DESCRIBE}"
    echo "upstream_tag=${GPU_TUNED_UPSTREAM_REF}"
    echo "variant=${GPU_TUNED_VARIANT}"
    echo "arch=sm_${GPU_TUNED_CUDA_ARCH}"
    echo "toolkit=${TOOLKIT_ID} (CUDA ${TOOLKIT_CUDA_VER})"
    echo "provenance=$(gpu_tuned_git_provenance "${REPODIR}")"
    echo "static_sha256=$(sha256sum "${STAGE}/${NAME}/lib/libnccl_static.a" | cut -d' ' -f1)"
} > "${STAGE}/${NAME}/BUILD_INFO"

TARBALL="${DIST_DIR}/${NAME}.tar.gz"
tar -C "${STAGE}" -czf "${TARBALL}" "${NAME}"
(cd "${DIST_DIR}" && sha256sum "${NAME}.tar.gz" > "${NAME}.tar.gz.sha256")
echo "Tarball: ${TARBALL} ($(du -sh "${TARBALL}" | awk '{print $1}'))"

RELEASE_NOTES="NCCL ${NCCL_TUNED_VERSION} built for ${GPU_TUNED_HW_LABEL}, single-arch (sm_${GPU_TUNED_CUDA_ARCH}), toolkit ${TOOLKIT_ID}. Static archive and headers for PyTorch's USE_SYSTEM_NCCL=1 USE_STATIC_NCCL=1. Based on NVIDIA/nccl ${GPU_TUNED_UPSTREAM_REF} plus the single-arch fix for NVIDIA/nccl issue 2420. See BUILD_INFO in the tarball for the exact commits."

if [[ -n "${NCCL_TUNED_NO_PUBLISH:-}" ]]; then
    printf '%s\n' "${RELEASE_TAG}" > "${DIST_DIR}/release-tag.txt"
    printf '%s\n' "${RELEASE_TITLE}" > "${DIST_DIR}/release-title.txt"
    printf '%s\n' "${RELEASE_NOTES}" > "${DIST_DIR}/release-notes.md"
    echo "NCCL_TUNED_NO_PUBLISH set: packaged ${RELEASE_TAG}, not published."
    exit 0
fi

gpu_tuned_publish_release "zbrad/nccl" "${RELEASE_TAG}" "${RELEASE_TITLE}" "${RELEASE_NOTES}" \
    "${TARBALL}#$(basename "${TARBALL}")" \
    "${TARBALL}.sha256#$(basename "${TARBALL}").sha256"
echo "Done: https://github.com/zbrad/nccl/releases/tag/${RELEASE_TAG}"
