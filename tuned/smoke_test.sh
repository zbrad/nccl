#!/bin/bash
# tuned/smoke_test.sh <variant> — check the tarball tuned/package.sh made, as a consumer receives it: extract it into a
# scratch directory and build and run tuned/tests/nccl_single_rank.cu against the EXTRACTED headers and static archive
# (not the build tree), after the checksum and identity checks. The package ships no shared library, so "the package's
# library is the one running" is shown by the test binary having no libnccl dependency at all (NCCL is linked in) and by
# the extracted archive being the one linked. A pass needs the end marker, so an abort cannot read as a pass.
#
# Writes <build root>/releases/SMOKE_RESULTS_<variant>.log; release.sh requires it newer than the tarball.
set -uo pipefail

REPODIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/smoke_test.sh <variant>}"
# shellcheck source=env.sh
source "${REPODIR}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}" || exit 1

DIST_DIR="${NCCL_TUNED_DIST_DIR}"
INFO="${DIST_DIR}/package-info.txt"
RELEASES_DIR="${NCCL_TUNED_BUILD_ROOT}/releases"
SMOKE_LOG="${RELEASES_DIR}/SMOKE_RESULTS_${GPU_TUNED_VARIANT}.log"
[[ -f "${INFO}" ]] || { echo "ERROR: ${INFO} not found -- run 'bash tuned/package.sh ${GPU_TUNED_VARIANT}' first." >&2; exit 1; }
info() { sed -n "s/^$1=//p" "${INFO}"; }
RELEASE_TAG="$(info release_tag)"; TARBALL="${DIST_DIR}/$(info tarball)"
mkdir -p "${RELEASES_DIR}"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
FAILURES=0
fail() { echo "  FAIL: $*" >&2; FAILURES=$((FAILURES + 1)); }
ok() { echo "  ok:   $*"; }

{
    echo "NCCL package smoke test -- variant=${GPU_TUNED_VARIANT} toolkit=${TOOLKIT_ID}"
    echo "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "release: ${RELEASE_TAG}"
    echo "tarball: $(basename "${TARBALL}")"
    echo ""
    if [[ ! -f "${TARBALL}" ]]; then
        fail "${TARBALL} not found"
    else
        [[ "$(sha256sum "${TARBALL}" | awk '{print $1}')" == "$(info sha256)" ]] && ok "sha256 matches package-info.txt" || fail "sha256 does not match package-info.txt"
        if gpu_tuned_verify_sha256_sidecar "${TARBALL}" 2>/dev/null; then ok "sha256 sidecar matches the tarball"; else fail "sha256 sidecar missing or not matching"; fi
        tar -xzf "${TARBALL}" -C "${WORK}" && ROOT="$(find "${WORK}" -mindepth 1 -maxdepth 1 -type d | head -1)" || ROOT=""
        if [[ -z "${ROOT}" ]]; then
            fail "the tarball does not extract"
        else
            ok "extracts to $(basename "${ROOT}")/"
            for f in lib/libnccl_static.a include/nccl.h LICENSE.txt BUILD_INFO BUILD_INFO.json; do
                [[ -s "${ROOT}/${f}" ]] || fail "the tarball is missing ${f}"
            done
            ls "${ROOT}"/lib/*.so* >/dev/null 2>&1 && fail "the tarball ships a shared library: $(ls "${ROOT}"/lib | tr '\n' ' ')" || ok "no shared library is shipped (static archive only)"

            # identity: the JSON build info names this release, variant and toolkit, and the header says the packaged version
            if gpu_tuned_verify_build_info_json_file "${ROOT}/BUILD_INFO.json" "release=${RELEASE_TAG}" "variant=${GPU_TUNED_VARIANT}" "cuda.tag=${TOOLKIT_ID}" 2>/dev/null; then
                ok "BUILD_INFO.json names ${RELEASE_TAG}, ${GPU_TUNED_VARIANT}, ${TOOLKIT_ID}"
            else
                fail "BUILD_INFO.json does not name ${RELEASE_TAG} / ${GPU_TUNED_VARIANT} / ${TOOLKIT_ID}"
            fi
            # the version string: <variant>-<toolkit>.<label> derived from the release tag must be what the packaged header and the
            # running library both say
            if [[ "${RELEASE_TAG}" =~ ^v0\.([0-9]{8})\.([0-9]+)-([a-z][a-z0-9]*)-(cu[0-9]+)$ ]]; then
                WANT_SUFFIX="-${BASH_REMATCH[3]}-${BASH_REMATCH[4]}.${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
                grep -q "^#define NCCL_SUFFIX \"${WANT_SUFFIX}\"" "${ROOT}/include/nccl.h" \
                    && ok "the packaged nccl.h defines NCCL_SUFFIX \"${WANT_SUFFIX}\" (from the tag)" || fail "the packaged nccl.h does not define NCCL_SUFFIX \"${WANT_SUFFIX}\""
            else
                WANT_SUFFIX=""; fail "release tag ${RELEASE_TAG} is not v0.<date>.<seq>-<variant>-<cuda>"
            fi
            HDR_VER="$(awk '$2=="NCCL_MAJOR"{M=$3} $2=="NCCL_MINOR"{N=$3} $2=="NCCL_PATCH"{P=$3} END{printf "%s.%s.%s", M, N, P}' "${ROOT}/include/nccl.h")"
            [[ "${HDR_VER}" == "$(info nccl_version | sed 's/[^0-9.].*//')" ]] && ok "the packaged header says NCCL ${HDR_VER}" || fail "the packaged header says ${HDR_VER}, package-info says $(info nccl_version)"
            export PATH="${CUDA_HOME}/bin:${PATH}"
            gpu_tuned_verify_arch "${ROOT}/lib/libnccl_static.a" "${GPU_TUNED_CUDA_ARCH}" >/dev/null 2>&1 \
                && ok "the extracted archive is single-arch sm_${GPU_TUNED_CUDA_ARCH}" || fail "the extracted archive is not single-arch sm_${GPU_TUNED_CUDA_ARCH}"

            # the test, compiled against the extracted package and run on the GPU
            if "${CUDA_HOME}/bin/nvcc" -O1 "-gencode=arch=compute_${GPU_TUNED_CUDA_ARCH},code=sm_${GPU_TUNED_CUDA_ARCH}" \
                -I "${ROOT}/include" -o "${WORK}/t_pkg" "${REPODIR}/tuned/tests/nccl_single_rank.cu" \
                -L "${ROOT}/lib" -l:libnccl_static.a -lrt -ldl -lpthread 2>"${WORK}/cc.log"; then
                if ldd "${WORK}/t_pkg" | grep -q libnccl; then fail "the test binary needs a shared libnccl: $(ldd "${WORK}/t_pkg" | grep libnccl)"; else ok "the test binary links the package's archive (no shared libnccl needed)"; fi
                if (cd "${WORK}" && NCCL_DEBUG=VERSION ./t_pkg) > "${WORK}/run.log" 2>&1 && grep -q '^PASSED [0-9]* checks$' "${WORK}/run.log"; then
                    ok "against the package: $(grep '^PASSED' "${WORK}/run.log")"
                    WANT_LINE="NCCL version $(info nccl_version)${WANT_SUFFIX}"
                    [[ "$(grep -m1 '^NCCL version' "${WORK}/run.log" || true)" == "${WANT_LINE}" ]] && ok "the packaged library reports '${WANT_LINE}'" || fail "the packaged library does not report exactly '${WANT_LINE}': $(grep -m1 'NCCL version' "${WORK}/run.log" || echo 'no version line')"
                else
                    fail "the test against the package: $(tail -3 "${WORK}/run.log" | tr '\n' ' ')"
                fi
            else
                fail "compile against the package: $(tail -3 "${WORK}/cc.log" | tr '\n' ' ')"
            fi
        fi
    fi
    echo ""
    if [[ ${FAILURES} -eq 0 ]]; then
        echo "  complete: every check ran"
        echo "=== Smoke test summary: passed ==="
    else
        echo "=== Smoke test summary: FAILED (${FAILURES}) ==="
    fi
    echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee "${SMOKE_LOG}"
echo "Results written to ${SMOKE_LOG}"
grep -q '^=== Smoke test summary: passed ===$' "${SMOKE_LOG}"
