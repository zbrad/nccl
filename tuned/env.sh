#!/bin/bash
# tuned/env.sh <variant> — single source of truth for a tuned NCCL build's CUDA
# toolkit selection and device metadata. Source this file with the variant as $1;
# do not execute it directly. Same GPU_TUNED_* convention as zbrad/llama.cpp,
# zbrad/raft, zbrad/cuvs and zbrad/faiss.
#
#   CUDA_TAG=cu133 bash tuned/build.sh gb10     (or CUDA_VER=13.3)
#
# The toolkit comes from the vendored definitions in tuned/common-data/toolkits/<id>.env
# (zbrad/tuned-common), never from /usr/local/cuda or whatever happens to be installed.
# With neither CUDA_TAG nor CUDA_VER set, the highest defined toolkit is used and the
# available list and the reason are printed. Repo-specific settings live in
# tuned/repo.env, loaded last; it may not change any shared key.

GPU_TUNED_ARG_VARIANT="${1:-}"
if [[ -z "${GPU_TUNED_ARG_VARIANT}" ]]; then
    echo "ERROR: env.sh requires a variant argument (gb10; rtx40/rtx50 not yet added)" >&2
    return 1 2>/dev/null || exit 1
fi

GPU_TUNED_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_TUNED_DATA_DIR="${GPU_TUNED_SELF_DIR}/common-data"
# shellcheck source=common-data/devices/gb10.conf
source "${GPU_TUNED_DATA_DIR}/devices/${GPU_TUNED_ARG_VARIANT}.conf" || return 1 2>/dev/null || exit 1
export GPU_TUNED_VARIANT GPU_TUNED_PLATFORM GPU_TUNED_CUDA_ARCH GPU_TUNED_HW_LABEL

# shellcheck source=common.sh
# Vendored from https://github.com/zbrad/tuned-common (see sync-common.sh).
source "${GPU_TUNED_SELF_DIR}/common.sh" || return 1 2>/dev/null || exit 1
gpu_tuned_assert_platform "${GPU_TUNED_PLATFORM}" "${GPU_TUNED_VARIANT}" || return 1 2>/dev/null || exit 1

# --- Build identity: NCCL's own version (makefiles/version.mk) and the commit. ---
REPODIR="$(cd "${GPU_TUNED_SELF_DIR}/.." && pwd)"
_nccl_ver_mk="${REPODIR}/makefiles/version.mk"
NCCL_TUNED_VERSION="$(awk '
    $1=="NCCL_MAJOR" {M=$3} $1=="NCCL_MINOR" {N=$3} $1=="NCCL_PATCH" {P=$3} $1=="NCCL_SUFFIX" {S=$3}
    END { if (M=="" || N=="" || P=="") exit 1; printf "%s.%s.%s%s", M, N, P, S }' "${_nccl_ver_mk}")" || {
    echo "[tuned/env] ERROR: cannot read NCCL_MAJOR/MINOR/PATCH from ${_nccl_ver_mk}." >&2
    return 1 2>/dev/null || exit 1
}
unset _nccl_ver_mk
: "${NCCL_TUNED_COMMIT:=$(git -C "${REPODIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)}"
# Source identity: NVIDIA's release tag, how many commits of ours sit on it, and the hash, e.g.
# v2.30.7-1-5-g2fe1779f (a bare upstream tag reads v2.30.7-1-0-g<hash>); -dirty when tracked
# files differ. Only NVIDIA's v<X.Y.Z>-<N> tags count: NCCL also keeps nccl4py-* and nccl-ep-*
# tags, and our own v0.* release tags must never be picked up.
NCCL_TUNED_DESCRIBE="$(git -C "${REPODIR}" describe --tags --long --dirty --match 'v[0-9]*.[0-9]*.[0-9]*-[0-9]*' --exclude 'v0.*' 2>/dev/null)" || {
    echo "[tuned/env] ERROR: git describe found no NVIDIA release tag (v<X.Y.Z>-<N>) in ${REPODIR}. Fetch upstream's tags: git fetch upstream --tags" >&2
    return 1 2>/dev/null || exit 1
}
NCCL_TUNED_COMMIT_FULL="$(git -C "${REPODIR}" rev-parse HEAD)"
[[ "${NCCL_TUNED_DESCRIBE}" == *-dirty ]] && NCCL_TUNED_COMMIT_FULL="${NCCL_TUNED_COMMIT_FULL}-dirty"
# NCCL prints NCCL_SUFFIX in its version line ("NCCL version 2.30.7-tuned+cuda13.4"), so a log
# from our build is recognisably not NVIDIA's. Passed on make's command line by build.sh, which
# overrides version.mk without editing that upstream-tracked file. Library file names and the
# SONAME use only the numbers, and nccl.h's version macros are unchanged.
NCCL_TUNED_SUFFIX="-tuned"
export NCCL_TUNED_VERSION NCCL_TUNED_COMMIT NCCL_TUNED_DESCRIBE NCCL_TUNED_COMMIT_FULL NCCL_TUNED_SUFFIX

# --- CUDA toolkit: chosen from tuned/common-data/toolkits. CUDA_TAG (cu133) or CUDA_VER
#     (13.3) select one; neither set means the highest defined, stated loudly. ---
_nccl_tk="${CUDA_TAG:-}"
_nccl_req_ver="${CUDA_VER:-}"
if [ -z "${_nccl_tk}" ] && [ -n "${_nccl_req_ver}" ]; then
    _nccl_tk="cu${_nccl_req_ver//./}"                           # 13.3 -> cu133
fi
gpu_tuned_load_toolkit "${GPU_TUNED_DATA_DIR}" "${GPU_TUNED_VARIANT}" "${_nccl_tk}" || return 1 2>/dev/null || exit 1
if [ -n "${_nccl_req_ver}" ] && [ "${_nccl_req_ver}" != "${TOOLKIT_CUDA_VER}" ]; then
    echo "[tuned/env] ERROR: CUDA_VER=${_nccl_req_ver} conflicts with toolkit ${TOOLKIT_ID} (CUDA ${TOOLKIT_CUDA_VER})." >&2
    return 1 2>/dev/null || exit 1
fi
unset _nccl_tk _nccl_req_ver
gpu_tuned_verify_toolkit || return 1 2>/dev/null || exit 1

# --- Output layout: build/<cuda>/<describe>/ (gitignored: /build*): one tree per toolkit and per
#     source state, so a build of a newer NCCL, or of one more commit of ours, never overwrites
#     an older one and each tree says what source made it. No variant level: two variants never
#     share a build tree. nccl/ is make's BUILDDIR, dist/ the tarballs. A marker file refuses a
#     tree last used by another variant, so a mix-up is an error, not a silently wrong archive. ---
NCCL_TUNED_BUILD_ROOT="${NCCL_TUNED_BUILD_ROOT:-${REPODIR}/build/${TOOLKIT_ID}/${NCCL_TUNED_DESCRIBE}}"
if [[ -f "${NCCL_TUNED_BUILD_ROOT}/.variant" && "$(cat "${NCCL_TUNED_BUILD_ROOT}/.variant")" != "${GPU_TUNED_VARIANT}" ]]; then
    echo "[tuned/env] ERROR: ${NCCL_TUNED_BUILD_ROOT} was built for variant '$(cat "${NCCL_TUNED_BUILD_ROOT}/.variant")', not '${GPU_TUNED_VARIANT}'. Use a separate checkout or NCCL_TUNED_BUILD_ROOT." >&2
    return 1 2>/dev/null || exit 1
fi
# ccache is per toolkit and shared by every source state of it (build/<cuda>/ccache, a sibling of
# the <describe> trees), so a rebuild of a newer commit reuses the device-code compiles that did
# not change, and timings and hits never cross toolkits. CCACHE_DIR also makes ccache ignore
# ~/.config/ccache/ccache.conf, so its small default max size would apply: set it here.
export CCACHE_DIR="${CCACHE_DIR:-${REPODIR}/build/${TOOLKIT_ID}/ccache}"
export CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-20G}"
NCCL_TUNED_BUILD_DIR="${NCCL_TUNED_BUILD_ROOT}/nccl"
NCCL_TUNED_DIST_DIR="${NCCL_TUNED_BUILD_ROOT}/dist"
export NCCL_TUNED_BUILD_ROOT NCCL_TUNED_BUILD_DIR NCCL_TUNED_DIST_DIR

# --- Repo-specific settings, loaded last; may not change any shared device/toolkit key. ---
gpu_tuned_load_repo_env "${GPU_TUNED_SELF_DIR}/repo.env" || return 1 2>/dev/null || exit 1

# embed_build_info <binary-or-so-path> — thin wrapper over gpu_tuned_embed_build_info
# (common.sh). Section name .nccl_tuned_build_info; the "version" field carries NCCL's own
# version, the commit and the toolkit tag. Not used on the static archive (an ELF section
# does not belong in a .a); the tarball's BUILD_INFO file carries the same fields.
embed_build_info() {
    local target="$1"
    gpu_tuned_embed_build_info "${target}" "${GPU_TUNED_VARIANT}" "nccl_tuned" \
        "${NCCL_TUNED_DESCRIBE}, ${TOOLKIT_ID}" \
        "${GPU_TUNED_HW_LABEL}" "https://github.com/zbrad/nccl"
}
