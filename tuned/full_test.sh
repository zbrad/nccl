#!/bin/bash
# tuned/full_test.sh <variant> — run the tuned NCCL functional tests against the BUILD TREE of one toolkit
# (tuned/build.sh output): tuned/tests/nccl_single_rank.cu, compiled twice, once against libnccl_static.a and once
# against the shared libnccl.so.2.x, and run on the GPU. NCCL has no upstream test suite in this repository, and this
# node has one GPU, so the suite is single-rank only: it does not exercise any multi-rank algorithm or transport
# (docs/ARCHITECTURE.md G3 records this as a decision; the header of the test says what it does prove).
#
# The shared run is made with LD_LIBRARY_PATH at the build tree and the binary's resolved libnccl is checked: it
# must be the build tree's file, never the system libnccl2.
#
# Writes <build root>/releases/TEST_RESULTS_<variant>.log; its last summary line is what release.sh requires.
#
#   CUDA_TAG=cu134 bash tuned/build.sh gb10 && CUDA_TAG=cu134 bash tuned/full_test.sh gb10
set -uo pipefail

REPODIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/full_test.sh <variant>}"
# shellcheck source=env.sh
source "${REPODIR}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}" || exit 1

BUILD_DIR="${NCCL_TUNED_BUILD_DIR}"
RELEASES_DIR="${NCCL_TUNED_BUILD_ROOT}/releases"
RESULTS="${RELEASES_DIR}/TEST_RESULTS_${GPU_TUNED_VARIANT}.log"
SHARED="$(ls "${BUILD_DIR}"/lib/libnccl.so.[0-9]*.[0-9]*.[0-9]* 2>/dev/null | head -1)"
[[ -f "${BUILD_DIR}/lib/libnccl_static.a" && -n "${SHARED}" && -f "${BUILD_DIR}/include/nccl.h" ]] || {
    echo "ERROR: ${BUILD_DIR} has no libnccl_static.a, libnccl.so.X.Y.Z and include/nccl.h. Run tuned/build.sh ${GPU_TUNED_VARIANT} first." >&2
    exit 1
}
mkdir -p "${RELEASES_DIR}"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
NVCC="${CUDA_HOME}/bin/nvcc"
GENCODE="-gencode=arch=compute_${GPU_TUNED_CUDA_ARCH},code=sm_${GPU_TUNED_CUDA_ARCH}"
SRC="${REPODIR}/tuned/tests/nccl_single_rank.cu"

FAILURES=0
fail() { echo "  FAIL: $*" >&2; FAILURES=$((FAILURES + 1)); }
ok() { echo "  ok:   $*"; }

{
    echo "NCCL full test suite -- variant=${GPU_TUNED_VARIANT} toolkit=${TOOLKIT_ID} (CUDA ${TOOLKIT_CUDA_VER})"
    echo "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "commit:  $(git -C "${REPODIR}" rev-parse HEAD)"
    echo "source:  ${NCCL_TUNED_DESCRIBE}"
    echo "build:   ${BUILD_DIR}"
    echo "scope:   single rank only (one GPU); no multi-rank algorithm or transport is exercised"
    echo ""

    # --- static ---
    if "${NVCC}" -O1 ${GENCODE} -I "${BUILD_DIR}/include" -o "${WORK}/t_static" "${SRC}" \
        -L "${BUILD_DIR}/lib" -l:libnccl_static.a -lrt -ldl -lpthread 2>"${WORK}/cc_static.log"; then
        if (cd "${WORK}" && ./t_static) > "${WORK}/run_static.log" 2>&1 && grep -q '^PASSED [0-9]* checks$' "${WORK}/run_static.log"; then
            ok "static: $(grep '^PASSED' "${WORK}/run_static.log")"
        else
            fail "static run: $(tail -3 "${WORK}/run_static.log" | tr '\n' ' ')"
        fi
        if ldd "${WORK}/t_static" | grep -q 'libnccl'; then fail "the static binary needs a shared libnccl: $(ldd "${WORK}/t_static" | grep libnccl)"; else ok "static binary has no shared libnccl dependency"; fi
    else
        fail "static compile: $(tail -3 "${WORK}/cc_static.log" | tr '\n' ' ')"
    fi

    # --- shared, from the build tree ---
    if "${NVCC}" -O1 ${GENCODE} -I "${BUILD_DIR}/include" -o "${WORK}/t_shared" "${SRC}" \
        -L "${BUILD_DIR}/lib" -lnccl 2>"${WORK}/cc_shared.log"; then
        RESOLVED="$(cd "${WORK}" && LD_LIBRARY_PATH="${BUILD_DIR}/lib" ldd ./t_shared | awk '/libnccl\.so/ {print $3}')"
        if [[ "$(readlink -f "${RESOLVED}")" == "$(readlink -f "${SHARED}")" ]]; then
            ok "shared binary resolves libnccl to the build tree (${RESOLVED##*/})"
        else
            fail "shared binary resolves libnccl to '${RESOLVED}', not the build tree's ${SHARED}"
        fi
        if (cd "${WORK}" && LD_LIBRARY_PATH="${BUILD_DIR}/lib" ./t_shared) > "${WORK}/run_shared.log" 2>&1 && grep -q '^PASSED [0-9]* checks$' "${WORK}/run_shared.log"; then
            ok "shared: $(grep '^PASSED' "${WORK}/run_shared.log")"
        else
            fail "shared run: $(tail -3 "${WORK}/run_shared.log" | tr '\n' ' ')"
        fi
    else
        fail "shared compile: $(tail -3 "${WORK}/cc_shared.log" | tr '\n' ' ')"
    fi

    echo ""
    if [[ ${FAILURES} -eq 0 ]]; then
        echo "  complete: every check ran"
        echo "=== Full test suite summary: passed ==="
    else
        echo "=== Full test suite summary: FAILED (${FAILURES}) ==="
    fi
    echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee "${RESULTS}"
STATUS=${PIPESTATUS[0]}
echo "Results written to ${RESULTS}"
grep -q '^=== Full test suite summary: passed ===$' "${RESULTS}"
