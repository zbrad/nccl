#!/bin/bash
# tuned/publish.sh <variant> — publishes the release that tuned/release.sh passed, as zbrad/nccl's GitHub release.
# It publishes nothing unless release.sh wrote the ready-marker, and only while HEAD, the notes and every asset are
# unchanged since then (gpu_tuned_publish_ready). The marker is then spent, so the same gates cannot publish twice.
#
#   CUDA_TAG=cu134 bash tuned/publish.sh gb10
set -uo pipefail

REPODIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="${1:-}"
case "${VARIANT}" in
    "") echo "usage: tuned/publish.sh <variant>" >&2; exit 2 ;;
    -*) echo "ERROR: unknown option ${VARIANT}" >&2; echo "usage: tuned/publish.sh <variant>" >&2; exit 2 ;;
esac
[[ $# -eq 1 ]] || { echo "ERROR: tuned/publish.sh takes one argument, the variant" >&2; exit 2; }

# shellcheck source=env.sh
source "${REPODIR}/tuned/env.sh" "${VARIANT}" || exit 1
cd "${REPODIR}"

MARKER="${NCCL_TUNED_BUILD_ROOT}/releases/RELEASE_READY_${VARIANT}"
gpu_tuned_publish_ready "${MARKER}" || exit 1
echo "Release: https://github.com/zbrad/nccl/releases/tag/$(sed -n 's/^tag=//p' "${MARKER}.published")"
