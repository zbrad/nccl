#!/bin/bash
# tuned/build.sh <variant> — build NCCL for one GPU variant only (single-arch), with the
# toolkit chosen in tuned/env.sh, into build/<cuda>/<describe>/nccl/ (gitignored: /build*): one
# tree per toolkit and source state, no variant level (see env.sh); build/<cuda>/<describe>/dist/
# holds the tarballs.
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

BUILD_DIR="${NCCL_TUNED_WORK_DIR}"          # where make runs (fixed path, so ccache hits across commits)
PUBLISH_DIR="${NCCL_TUNED_BUILD_DIR}"       # build/<cuda>/<describe>/nccl: what package.sh reads
JOBS="${NCCL_TUNED_JOBS:-$(gpu_tuned_jobs)}"
GENCODE="-gencode=arch=compute_${GPU_TUNED_CUDA_ARCH},code=sm_${GPU_TUNED_CUDA_ARCH}"

# The fix must be present: without it NCCL adds sm_100 device code whatever NVCC_GENCODE says.
if ! grep -q "zbrad/nccl tuned-builds: version line" "${REPODIR}/src/init.cc"; then
    echo "ERROR: src/init.cc does not carry the version-line patch (marker 'zbrad/nccl tuned-builds: version line'). Check the last merge." >&2
    exit 1
fi
if ! grep -q "zbrad/nccl tuned-builds: compile LDMC" "${REPODIR}/src/device/Makefile"; then
    echo "ERROR: src/device/Makefile does not carry the single-arch fix (marker 'zbrad/nccl tuned-builds: compile LDMC')." >&2
    echo "  Without it the build embeds sm_100 code whatever NVCC_GENCODE says. Check the last merge." >&2
    exit 1
fi

# The release label is baked into the library's version string, so it is allocated here, once, and recorded for the
# other stages (env.sh reads it). A rebuild of the same tree keeps it; NCCL_TUNED_RELABEL=1 allocates a new one (for
# example when its sequence has since been published). GPU_TUNED_LABEL_SEQ=<n> chooses the sequence offline.
if [[ -z "${NCCL_TUNED_LABEL}" || -n "${NCCL_TUNED_RELABEL:-}" ]]; then
    NCCL_TUNED_LABEL="$(gpu_tuned_allocate_date_label "${REPODIR}")" || exit 1
    mkdir -p "${NCCL_TUNED_BUILD_ROOT}"
    printf '%s\n' "${NCCL_TUNED_LABEL}" > "${NCCL_TUNED_LABEL_FILE}"
    NCCL_TUNED_SUFFIX="-${GPU_TUNED_VARIANT}-${TOOLKIT_ID}.${NCCL_TUNED_LABEL%%.*}.$(gpu_tuned_label_seq "${NCCL_TUNED_LABEL}")"
    export NCCL_TUNED_LABEL NCCL_TUNED_SUFFIX
    echo "Release label allocated: ${NCCL_TUNED_LABEL} (version string ${NCCL_TUNED_VERSION}${NCCL_TUNED_SUFFIX})"
fi

echo "===================================================="
echo "NCCL ${NCCL_TUNED_DESCRIBE} ${GPU_TUNED_HW_LABEL} build"
echo "===================================================="
echo "  Toolkit   : ${TOOLKIT_ID} (${CUDA_HOME})"
echo "  Gencode   : ${GENCODE}"
echo "  Work dir  : ${BUILD_DIR}"
echo "  Output    : ${PUBLISH_DIR}"
echo "  Jobs      : ${JOBS}"
echo ""

# NCCL's make does not track the variables passed here (gencode, suffix, git hash, toolkit) as
# dependencies, so a rebuild with different values silently reuses stale objects and a stale
# generated nccl.h. Record them, plus a hash of any uncommitted changes, and start from an empty
# tree when they differ. Deleting only ever happens under this repo's build/ directory.
PARAMS="gencode=${GENCODE} suffix=${NCCL_TUNED_SUFFIX} commit=${NCCL_TUNED_COMMIT_FULL} toolkit=${TOOLKIT_ID} dirty=$(git -C "${REPODIR}" diff HEAD -- . ':!build' | sha256sum | cut -d' ' -f1)"
if [[ -f "${BUILD_DIR}/.params" && "$(cat "${BUILD_DIR}/.params")" != "${PARAMS}" ]]; then
    [[ "${BUILD_DIR}" == "${REPODIR}"/build/*/work/nccl ]] || { echo "ERROR: refusing to clear unexpected build dir ${BUILD_DIR}" >&2; exit 1; }
    echo "Build parameters changed since the last build here; starting from an empty tree."
    rm -rf -- "${BUILD_DIR}"
fi
# Compiler cache. nvcc is wrapped directly (that is where most of the build time is: the generated
# device kernels). Host C++ is routed through ccache by tuned/timed-shell.sh, because CXX itself must
# stay a bare compiler (NCCL passes "-ccbin $(CXX)" to nvcc). nvcc -dw compiles (the symmetric
# kernels) are not cacheable: ccache does not recognise -dw as a compile. NCCL_TUNED_NO_CCACHE=1
# builds without any of it.
NVCC_CMD="${CUDA_HOME}/bin/nvcc"
if command -v ccache >/dev/null 2>&1 && [[ -z "${NCCL_TUNED_NO_CCACHE:-}" ]]; then
    NVCC_CMD="ccache ${NVCC_CMD}"
    mkdir -p "${CCACHE_DIR}"
    # Host C++ goes through ccache too, via tuned/timed-shell.sh and a `c++` symlink to ccache.
    NCCL_TUNED_CCACHE_BIN="$(dirname "${CCACHE_DIR}")/ccache-bin"
    mkdir -p "${NCCL_TUNED_CCACHE_BIN}"
    ln -sfn "$(command -v ccache)" "${NCCL_TUNED_CCACHE_BIN}/c++"
    export NCCL_TUNED_CCACHE_BIN
    ccache -z >/dev/null
    echo "  ccache    : ${CCACHE_DIR} (max ${CCACHE_MAXSIZE})"
fi
mkdir -p "${BUILD_DIR}" "${NCCL_TUNED_BUILD_ROOT}"
printf '%s\n' "${PARAMS}" > "${BUILD_DIR}/.params"
printf '%s\n' "${GPU_TUNED_VARIANT}" > "${NCCL_TUNED_BUILD_ROOT}/.variant"

MAKE_ARGS=(
    -C "${REPODIR}"
    CXX="${CXX:-c++}"
    CUDA_HOME="${CUDA_HOME}"
    NVCC="${NVCC_CMD}"
    NVCC_GENCODE="${GENCODE}"
    BUILDDIR="${BUILD_DIR}"
    NCCL_SUFFIX="${NCCL_TUNED_SUFFIX}"
    NCCL_GIT_BRANCH="$(git -C "${REPODIR}" rev-parse --abbrev-ref HEAD)"
    NCCL_GIT_COMMIT_HASH="${NCCL_TUNED_COMMIT_FULL}"
    VERBOSE=0 DEBUG=0
)

# Progress and timing. NCCL's make prints a "Compiling <file>" line when a unit starts and nothing
# else, so tuned/timed-shell.sh (make's SHELL) times every recipe line that names an output with
# "-o <path>" (each compile and link) and prints "[done/total t+elapsed] seconds kind name". The
# total is the number of such lines a `make -n` dry run reports (on a partly built tree, only the
# remaining ones). Per-unit results stay in
# build/<cuda>/<describe>/timing.tsv: seconds, kind, name, exit status.
UNIT_RE='[[:space:]]-o[[:space:]]+[^[:space:];]+'
TOTAL="$(make -n "${MAKE_ARGS[@]}" 2>/dev/null | grep -cE "${UNIT_RE}" || true)"
export NCCL_TUNED_STRIP_PREFIX="${BUILD_DIR}"
export NCCL_TUNED_TIMING_FILE="${NCCL_TUNED_BUILD_ROOT}/timing.tsv"
export NCCL_TUNED_TIMING_TOTAL="${TOTAL}"
export NCCL_TUNED_TIMING_T0="${EPOCHSECONDS}"
: > "${NCCL_TUNED_TIMING_FILE}"
echo "  Units     : ${TOTAL} to build (timings: ${NCCL_TUNED_TIMING_FILE})"
echo ""
make "${MAKE_ARGS[@]}" -j"${JOBS}" -l"${JOBS}" SHELL="${REPODIR}/tuned/timed-shell.sh"

WALL=$(( EPOCHSECONDS - NCCL_TUNED_TIMING_T0 ))
echo ""
awk -F'\t' -v wall="${WALL}" -v total="${TOTAL}" '
    { n++; cpu += $1; if ($4 != 0) bad++ }
    END { printf "Timing: %d units timed (of %d expected), wall %ds, summed unit time %.0fs, %d failed\n", n, total, wall, cpu, bad }
' "${NCCL_TUNED_TIMING_FILE}"
echo "Slowest units:"
sort -t$'\t' -k1,1 -rn "${NCCL_TUNED_TIMING_FILE}" | head -10 | awk -F'\t' '{ printf "  %7.1fs  %-10s %s\n", $1, $2, $3 }'
if [[ "${NVCC_CMD}" == ccache* ]]; then
    echo "ccache: $(ccache -s | grep -E 'Hits:|Misses:' | head -2 | tr -s ' ' | tr '\n' ' ')"
fi

SHARED="$(find "${BUILD_DIR}/lib" -maxdepth 1 -name 'libnccl.so.*.*.*' -not -type l | head -1)"
STATIC="${BUILD_DIR}/lib/libnccl_static.a"
[[ -f "${SHARED}" && -f "${STATIC}" ]] || { echo "ERROR: build produced no ${BUILD_DIR}/lib/libnccl.so.X.Y.Z and libnccl_static.a" >&2; exit 1; }

echo ""
if ! grep -q "^#define NCCL_SUFFIX \"${NCCL_TUNED_SUFFIX}\"" "${BUILD_DIR}/include/nccl.h"; then
    echo "ERROR: ${BUILD_DIR}/include/nccl.h does not define NCCL_SUFFIX \"${NCCL_TUNED_SUFFIX}\"." >&2
    exit 1
fi
# (no pipe into grep -q here: under pipefail grep's early exit makes strings die of SIGPIPE and
#  the check would fail although the line is present)
if ! grep -qx "NCCL version ${NCCL_TUNED_VERSION}${NCCL_TUNED_SUFFIX}" <(strings -n 12 "${SHARED}"); then
    echo "ERROR: ${SHARED} does not report exactly 'NCCL version ${NCCL_TUNED_VERSION}${NCCL_TUNED_SUFFIX}' as its version line (no '+cuda' tail)." >&2
    exit 1
fi
echo "ok:   NCCL reports itself as ${NCCL_TUNED_VERSION}${NCCL_TUNED_SUFFIX}"
echo "Verifying single-arch (sm_${GPU_TUNED_CUDA_ARCH})..."
export PATH="${CUDA_HOME}/bin:${PATH}"
gpu_tuned_verify_arch "${SHARED}" "${GPU_TUNED_CUDA_ARCH}"
gpu_tuned_verify_arch "${STATIC}" "${GPU_TUNED_CUDA_ARCH}"
echo ""
# Publish the finished library and headers into the <describe> tree.
[[ "${PUBLISH_DIR}" == "${REPODIR}"/build/*/*/nccl && "${PUBLISH_DIR}" != */work/nccl ]] || { echo "ERROR: unexpected output dir ${PUBLISH_DIR}" >&2; exit 1; }
mkdir -p "${PUBLISH_DIR}"
rm -rf -- "${PUBLISH_DIR}/lib" "${PUBLISH_DIR}/include"
cp -a "${BUILD_DIR}/lib" "${BUILD_DIR}/include" "${PUBLISH_DIR}/"
printf '%s\n' "${PARAMS}" > "${PUBLISH_DIR}/.params"
echo "Built: ${PUBLISH_DIR}/lib ($(basename "${SHARED}"), libnccl_static.a)"
echo "Next: CUDA_TAG=${TOOLKIT_ID} bash tuned/release.sh ${GPU_TUNED_VARIANT}"
