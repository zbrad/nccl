# tuned/ — single-arch NCCL builds

Builds NCCL for one GPU variant and one CUDA toolkit, compiling device code for exactly that variant's architecture
(`NVCC_GENCODE`), the same `make` invocation PyTorch's own `cmake/External/nccl.cmake` runs when it builds NCCL in-tree.
The shared library and the static archive are both checked for that one architecture. The tuned builds of zbrad/pytorch and zbrad/llama.cpp
pin one of these releases as their NCCL and do not use the system `libnccl2` (the raft-dask Python wheel is the exception:
it links the pip `nvidia-nccl-cu13` wheel). The layout and rules are
`docs/ARCHITECTURE.md` in [zbrad/tuned-common](https://github.com/zbrad/tuned-common). Only gb10 is built here today.

## One branch per toolkit

A release for a toolkit is made from that toolkit's branch, which is based on the NCCL release NVIDIA ships for that CUDA:
`tuned-builds-cu133` and `tuned-builds-cu134` (`tuned-builds` is the common base). A change to `tuned/` goes on each
branch it applies to. Check the branch before building or releasing.

```
CUDA_TAG=cu134 bash tuned/build.sh gb10        # NCCL, single-arch               -> build/cu134/nccl/
CUDA_TAG=cu134 bash tuned/full_test.sh gb10    # single-rank functional tests    -> build/cu134/releases/ log
CUDA_TAG=cu134 bash tuned/package.sh gb10      # tarball, sha256, DRAFT notes    -> build/cu134/dist/
CUDA_TAG=cu134 bash tuned/smoke_test.sh gb10
# review and commit tuned/releases/RELEASE_NOTES_<label>-<variant>-<cuda>.md
CUDA_TAG=cu134 bash tuned/release.sh gb10      # gates only; writes the ready-marker, never publishes
CUDA_TAG=cu134 bash tuned/publish.sh gb10      # publishes exactly what release.sh saw
```

Unlike the other tuned repos the build tree is `build/<toolkit>/` at the repo root (gitignored), not `tuned/build/`.

## Files

| File | Purpose |
|---|---|
| `env.sh` | device, toolkit and tree setup; sourced by every script |
| `repo.env` | NCCL-only settings |
| `common.sh`, `sync-common.sh`, `common-data/` | vendored from tuned-common; refresh with `bash tuned/sync-common.sh sync` |
| `build.sh`, `full_test.sh`, `package.sh`, `smoke_test.sh`, `release.sh`, `publish.sh` | the stages above |
| `tests/nccl_single_rank.cu` | the functional test, compiled twice (against `libnccl_static.a` and the shared library) and run on the GPU |
| `timed-shell.sh` | a `make SHELL` that times every recipe line of the build; `docs/build_timings.md` and `docs/build_timings/` hold the records |
| `releases/` | tracked release notes |
| `compliance.conf` | attestations for the checks `tools/tuned_compliance.sh` cannot make mechanically |

## Identity

- The release tag is `v0.<YYYYMMDD>.<seq>-<variant>-<cuda>` (for example `v0.20261018.0-gb10-cu134`), allocated by
  `package.sh` and read from `package-info.txt` by `release.sh`. NCCL's own `v2.x.y-n` tags are matched for the version and
  `v0.*` tags are excluded from that lookup.
- The library's version string carries the same identity: `NCCL version <NCCL version>-<variant>-<toolkit>.<label>`, with
  upstream's `+cuda<major>.<minor>` tail removed by a one-line patch in `src/init.cc` (marker `zbrad/nccl tuned-builds:
  version line`).

## Tests

The suite is single-rank only. There is no upstream test suite in this repository and the build node has one GPU, so no
multi-rank algorithm or transport is exercised (recorded in `docs/ARCHITECTURE.md` as decision G3). The shared run puts the
build tree's library first and checks that the binary's resolved `libnccl` is that file and not the system library.

There is no Python in this build, so no build venv and none of the Python dependency gates.
