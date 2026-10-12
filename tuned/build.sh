#!/bin/bash
# tuned/build.sh <variant> — build NCCL for one GPU variant only (single-arch), with the
# toolkit chosen in tuned/env.sh, into build/<cuda>/<variant>/ (gitignored: /build*), the
# layout tuned-common's gpu_tuned_cuda_subdir gives every repo (cuda tag first, then variant).
#
#   CUDA_TAG=cu133 bash tuned/build.sh gb10
#
# The make invocation is the one PyTorch's own cmake/External/nccl.cmake runs when it builds
# NCCL in-tree, with NVCC_GENCODE for exactly the variant's arch, so the result is what torch
# links today -- minus the in-tree build. The single-arch fix in src/device/Makefile
# (NVIDIA/nccl issue 2420) is what makes NVCC_GENCODE the only target. Afterwards the shared
# library and the static archive are both checked for exactly that one arch.
set -euo pipefail

REPODIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/build.sh <variant>}"
# shellcheck source=env.sh
source "${REPODIR}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}" || exit 1

BUILD_DIR="${NCCL_TUNED_BUILD_DIR:-$(gpu_tuned_cuda_subdir "${REPODIR}/build" "${TOOLKIT_ID}" "${GPU_TUNED_VARIANT}")}"
JOBS="${NCCL_TUNED_JOBS:-$(nproc)}"
GENCODE="-gencode=arch=compute_${GPU_TUNED_CUDA_ARCH},code=sm_${GPU_TUNED_CUDA_ARCH}"

# The fix must be present: without it NCCL adds sm_100 device code whatever NVCC_GENCODE says.
if ! grep -q "zbrad/nccl tuned-builds: compile LDMC" "${REPODIR}/src/device/Makefile"; then
    echo "ERROR: src/device/Makefile does not carry the single-arch fix (marker 'zbrad/nccl tuned-builds: compile LDMC')." >&2
    echo "  Without it the build embeds sm_100 code whatever NVCC_GENCODE says. Check the last merge." >&2
    exit 1
fi

echo "===================================================="
echo "NCCL ${NCCL_TUNED_VERSION} (${NCCL_TUNED_COMMIT}) ${GPU_TUNED_HW_LABEL} build"
echo "===================================================="
echo "  Toolkit   : ${TOOLKIT_ID} (${CUDA_HOME})"
echo "  Gencode   : ${GENCODE}"
echo "  Build dir : ${BUILD_DIR}"
echo "  Jobs      : ${JOBS}"
echo ""

mkdir -p "${BUILD_DIR}"
make -C "${REPODIR}" -j"${JOBS}" -l"${JOBS}" \
    CXX="${CXX:-c++}" \
    CUDA_HOME="${CUDA_HOME}" \
    NVCC="${CUDA_HOME}/bin/nvcc" \
    NVCC_GENCODE="${GENCODE}" \
    BUILDDIR="${BUILD_DIR}" \
    VERBOSE=0 DEBUG=0

SHARED="$(find "${BUILD_DIR}/lib" -maxdepth 1 -name 'libnccl.so.*.*.*' -not -type l | head -1)"
STATIC="${BUILD_DIR}/lib/libnccl_static.a"
[[ -f "${SHARED}" && -f "${STATIC}" ]] || { echo "ERROR: build produced no ${BUILD_DIR}/lib/libnccl.so.X.Y.Z and libnccl_static.a" >&2; exit 1; }

echo ""
echo "Verifying single-arch (sm_${GPU_TUNED_CUDA_ARCH})..."
export PATH="${CUDA_HOME}/bin:${PATH}"
gpu_tuned_verify_arch "${SHARED}" "${GPU_TUNED_CUDA_ARCH}"
gpu_tuned_verify_arch "${STATIC}" "${GPU_TUNED_CUDA_ARCH}"
echo ""
echo "Built: ${BUILD_DIR}/lib ($(basename "${SHARED}"), libnccl_static.a)"
echo "Next: CUDA_TAG=${TOOLKIT_ID} bash tuned/release.sh ${GPU_TUNED_VARIANT}"
