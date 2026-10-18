#!/bin/bash
# tuned/env.sh <variant> — single source of truth for a tuned NCCL build's CUDA
# toolkit selection and device metadata. Source this file with the variant as $1;
# do not execute it directly. Same GPU_TUNED_* convention as zbrad/llama.cpp,
# zbrad/raft, zbrad/cuvs and zbrad/faiss.
#
#   CUDA_TAG=cu133 bash tuned/build.sh gb10     (or CUDA_VER=13.3)
#
# The toolkit comes from the vendored definitions in tuned/common-data/toolkits/<id>.env
# (zbrad/tuned-common), never from /usr/local/cuda or whatever happens to be installed. There is no
# default: CUDA_TAG or CUDA_VER must name one; with neither set the loader fails and lists the
# defined toolkits (at a terminal it asks, for that run only). Repo-specific settings live in
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
export NCCL_TUNED_VERSION NCCL_TUNED_COMMIT NCCL_TUNED_DESCRIBE NCCL_TUNED_COMMIT_FULL

# --- CUDA toolkit: named by CUDA_TAG (cu133) or CUDA_VER (13.3), never defaulted or guessed; loaded from
#     tuned/common-data/toolkits and verified (docs/ARCHITECTURE.md B4). Exports CUDA_VER, CUDA_TAG, CUDA_HOME. ---
gpu_tuned_select_toolkit "${GPU_TUNED_DATA_DIR}" "${GPU_TUNED_VARIANT}" || return 1 2>/dev/null || exit 1

# --- Output layout (docs/ARCHITECTURE.md C2): build/<cuda>/ (gitignored: /build*), one tree per toolkit and no
#     variant level or per-commit level:
#       ccache/ ccache-bin/   per toolkit        work/nccl/   where make runs (objects), a fixed path
#       nccl/                 the published lib/ and include/ of the last build, with .params (the exact source and settings)
#       dist/                 tarballs and package-info.txt          releases/   test and smoke logs
#       release-label  timing.tsv  .variant
#     Variants are never built beside each other in the same storage; a marker file refuses a tree last used by another
#     variant, so a mix-up is an error, not a silently wrong archive. Which source a build came from is recorded in
#     nccl/.params and checked by nccl_tuned_assert_build_current below, not implied by a directory name. ---
NCCL_TUNED_BUILD_ROOT="${NCCL_TUNED_BUILD_ROOT:-${REPODIR}/build/${TOOLKIT_ID}}"
if [[ -f "${NCCL_TUNED_BUILD_ROOT}/.variant" && "$(cat "${NCCL_TUNED_BUILD_ROOT}/.variant")" != "${GPU_TUNED_VARIANT}" ]]; then
    echo "[tuned/env] ERROR: ${NCCL_TUNED_BUILD_ROOT} was built for variant '$(cat "${NCCL_TUNED_BUILD_ROOT}/.variant")', not '${GPU_TUNED_VARIANT}'. Use a separate checkout or NCCL_TUNED_BUILD_ROOT." >&2
    return 1 2>/dev/null || exit 1
fi
# ccache is per toolkit (build/<cuda>/ccache), so a rebuild of a newer commit reuses the device-code
# compiles that did not change, and timings and hits never cross toolkits. CCACHE_DIR also makes ccache ignore
# ~/.config/ccache/ccache.conf, so its small default max size would apply: set it here.
export CCACHE_DIR="${CCACHE_DIR:-${REPODIR}/build/${TOOLKIT_ID}/ccache}"
export CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-20G}"
# The compile itself happens in ONE fixed directory per toolkit, build/<cuda>/work/nccl, and the
# finished lib/ and include/ are published into build/<cuda>/nccl. ccache keys on the paths in
# the compile command lines (-o, -I, the generated sources), so building in a directory named after
# the commit would make every new commit a full cache miss; with a fixed work path a new commit
# reuses every unit it did not change.
NCCL_TUNED_WORK_DIR="${NCCL_TUNED_WORK_DIR:-${REPODIR}/build/${TOOLKIT_ID}/work/nccl}"
NCCL_TUNED_BUILD_DIR="${NCCL_TUNED_BUILD_ROOT}/nccl"
NCCL_TUNED_DIST_DIR="${NCCL_TUNED_BUILD_ROOT}/dist"
export NCCL_TUNED_BUILD_ROOT NCCL_TUNED_BUILD_DIR NCCL_TUNED_WORK_DIR NCCL_TUNED_DIST_DIR

# nccl_tuned_built_commit -- the full commit the published build in build/<cuda>/nccl was made from ("<sha>" or "<sha>-dirty"), read from
# its .params; empty if there is no build. nccl_tuned_describe_commit <sha> -- NVIDIA's release tag, our commits on it and the hash.
nccl_tuned_built_commit() { sed -n 's/.* commit=\([0-9a-f]\{40\}\(-dirty\)\{0,1\}\) .*/\1/p' "${NCCL_TUNED_BUILD_DIR}/.params" 2>/dev/null | head -1; }
nccl_tuned_describe_commit() { git -C "${REPODIR}" describe --tags --long --match 'v[0-9]*.[0-9]*.[0-9]*-[0-9]*' --exclude 'v0.*' "$1" 2>/dev/null; }
# nccl_tuned_assert_build_current -- fail unless the published build is clean and was made from source that equals HEAD outside tuned/
# (docs/ARCHITECTURE.md H6). Replaces the per-commit directory that used to make a stale build impossible to pick up by path.
nccl_tuned_assert_build_current() {
    local c changed
    c="$(nccl_tuned_built_commit)"
    [[ -n "${c}" ]] || { echo "ERROR: ${NCCL_TUNED_BUILD_DIR}/.params records no build commit: run tuned/build.sh ${GPU_TUNED_VARIANT}." >&2; return 1; }
    [[ "${c}" != *-dirty ]] || { echo "ERROR: the build in ${NCCL_TUNED_BUILD_DIR} was made from a dirty tree (${c}); commit and rebuild." >&2; return 1; }
    git -C "${REPODIR}" cat-file -e "${c}^{commit}" 2>/dev/null || { echo "ERROR: the build was made at ${c}, which is not in this checkout." >&2; return 1; }
    changed="$(git -C "${REPODIR}" diff --name-only "${c}" HEAD -- . ':!tuned' | head -5)"
    [[ -z "${changed}" ]] || { echo "ERROR: the build was made at ${c:0:9} but source outside tuned/ changed since: $(echo "${changed}" | tr '\n' ' '): rebuild." >&2; return 1; }
}

# --- The version NCCL reports: NCCL prints NCCL_SUFFIX in its version line, so the library says what it is:
#       NCCL version 2.32.3-gb10-cu134.20261018.0
#     (upstream appends "+cuda<major>.<minor>" to this line; this fork drops it in src/init.cc, the suffix already names the toolkit)
#     i.e. <NCCL version>-<variant>-<toolkit>.<release label>, the same identity as the tag (v0.20261018.0-gb10-cu134) and
#     the tarball. The label is allocated ONCE, by tuned/build.sh, because it is compiled into nccl.h and the library
#     (docs/ARCHITECTURE.md E4), and recorded in <build root>/release-label; every other script reads it here and never
#     recomputes it. Before the first build there is no label, and the suffix is only -<variant>-<toolkit>.
#     Passed on make's command line by build.sh, which overrides version.mk without editing that upstream-tracked file.
#     Library file names and the SONAME use only the numbers, and nccl.h's version macros are unchanged. ---
NCCL_TUNED_LABEL_FILE="${NCCL_TUNED_BUILD_ROOT}/release-label"
NCCL_TUNED_LABEL="$(cat "${NCCL_TUNED_LABEL_FILE}" 2>/dev/null || true)"      # <YYYYMMDD> or <YYYYMMDD>.<seq>, as the allocator prints it
if [[ -n "${NCCL_TUNED_LABEL}" ]]; then
    NCCL_TUNED_SUFFIX="-${GPU_TUNED_VARIANT}-${TOOLKIT_ID}.${NCCL_TUNED_LABEL%%.*}.$(gpu_tuned_label_seq "${NCCL_TUNED_LABEL}")"
else
    NCCL_TUNED_SUFFIX="-${GPU_TUNED_VARIANT}-${TOOLKIT_ID}"
fi
export NCCL_TUNED_LABEL_FILE NCCL_TUNED_LABEL NCCL_TUNED_SUFFIX

# --- Repo-specific settings, loaded last; may not change any shared device/toolkit key. ---
gpu_tuned_load_repo_env "${GPU_TUNED_SELF_DIR}/repo.env" || return 1 2>/dev/null || exit 1
