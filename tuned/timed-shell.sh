#!/bin/bash
# tuned/timed-shell.sh -c '<recipe>' — a make SHELL that runs each recipe line normally and records
# how long it took. tuned/build.sh passes it as `make SHELL=...`, so every compile and link, host and
# device, is timed without touching the compiler command lines (NCCL hands nvcc "-ccbin $(CXX)", so
# CXX cannot be wrapped).
#
# NCCL's recipes are several lines each (a message, mkdir, the compiler, a dependency step), and
# make runs every line through SHELL separately, so the message line is not the work. A recipe line
# is a unit when it names an output with "-o <path>": every compile and link does, the
# dependency-generation and mkdir lines do not. Units are timed, appended to
# $NCCL_TUNED_TIMING_FILE as
#   <seconds>\t<kind>\t<output>\t<exit status>
# and reported as "[done/total t+elapsed] <seconds>s <kind> <output>". Other lines just run.
# kind is nvcc, cxx, or link (a -shared library or an executable).
#
# Environment (set by build.sh): NCCL_TUNED_TIMING_FILE, NCCL_TUNED_TIMING_TOTAL (units expected;
# 0 = unknown), NCCL_TUNED_TIMING_T0 (epoch seconds when the build started), NCCL_TUNED_STRIP_PREFIX
# (stripped from the front of output names). Only `-c <recipe>` is supported, which is how GNU
# make invokes SHELL.

if [[ "${1:-}" != "-c" || $# -ne 2 ]]; then
    echo "timed-shell.sh: expected exactly '-c <recipe>', got $# argument(s)" >&2
    exit 2
fi
recipe="$2"

# The same predicate build.sh uses to count units in a `make -n` dry run.
UNIT_RE='[[:space:]]-o[[:space:]]+([^[:space:];]+)'

# Host C++ through ccache. NCCL passes "-ccbin $(CXX)" to nvcc, so CXX must stay a bare compiler and
# cannot be "ccache c++". Instead, when NCCL_TUNED_CCACHE_BIN names a directory holding a `c++`
# symlink to ccache, only recipes that do not run nvcc get it first on PATH: nvcc's own host-compiler
# calls are left alone. NCCL also puts -DNCCL_GIT_COMMIT_HASH/-DNCCL_GIT_BRANCH on EVERY C++ command
# line, which would make every compile miss after any new commit; only git_version.cc reads them, so
# they are removed from every other recipe.
if [[ -n "${NCCL_TUNED_CCACHE_BIN:-}" && "${recipe}" != *nvcc* && "${recipe}" == *" -c "* ]]; then
    if [[ "${recipe}" != *git_version* ]]; then
        recipe="$(printf '%s' "${recipe}" | sed -E "s/-DNCCL_GIT_(COMMIT_HASH|BRANCH)='\"[^\"]*\"'//g")"
    fi
    export PATH="${NCCL_TUNED_CCACHE_BIN}:${PATH}"
fi

if [[ -z "${NCCL_TUNED_TIMING_FILE:-}" || ! "${recipe}" =~ ${UNIT_RE} ]]; then
    exec /bin/bash -c "${recipe}"
fi
name="${BASH_REMATCH[1]}"
name="${name#"${NCCL_TUNED_STRIP_PREFIX:-/nonexistent}"/}"
if [[ "${recipe}" == *nvcc* ]]; then kind=nvcc
elif [[ "${recipe}" == *" -shared "* ]]; then kind=link
elif [[ "${recipe}" == *" -c "* ]]; then kind=cxx
else kind=link; fi

start="${EPOCHREALTIME}"
/bin/bash -c "${recipe}"
rc=$?
end="${EPOCHREALTIME}"

secs="$(awk -v a="${start}" -v b="${end}" 'BEGIN { printf "%.1f", b - a }')"
# Append and count under one lock, so parallel jobs that finish together get distinct numbers.
done_n="$(
    exec 9>>"${NCCL_TUNED_TIMING_FILE}.lock"
    flock 9
    printf '%s\t%s\t%s\t%s\n' "${secs}" "${kind}" "${name}" "${rc}" >> "${NCCL_TUNED_TIMING_FILE}"
    wc -l < "${NCCL_TUNED_TIMING_FILE}"
)"
total="${NCCL_TUNED_TIMING_TOTAL:-0}"
elapsed=""
if [[ -n "${NCCL_TUNED_TIMING_T0:-}" ]]; then
    elapsed=" t+$(( EPOCHSECONDS - NCCL_TUNED_TIMING_T0 ))s"
fi
if [[ "${total}" -gt 0 ]]; then
    printf '[%d/%d%s] %ss %s %s\n' "${done_n}" "${total}" "${elapsed}" "${secs}" "${kind}" "${name}"
else
    printf '[%d%s] %ss %s %s\n' "${done_n}" "${elapsed}" "${secs}" "${kind}" "${name}"
fi
exit "${rc}"
