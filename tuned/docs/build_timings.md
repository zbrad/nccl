# NCCL tuned build timings

How long `tuned/build.sh` takes for a single-arch NCCL build, what the compiler cache saves, and
what it cannot. Everything here was measured; the per-unit data is in
[`build_timings/`](build_timings/) and the commands to repeat it are below.

## Setup

| | |
|---|---|
| Machine | `geo-a-node-gb10-1`: aarch64, 20 cores (`nproc`), jobs = 20 |
| Variant / arch | gb10, `-gencode=arch=compute_121a,code=sm_121a` (single arch) |
| Toolkit | CUDA 13.4 (`cu134`), ccache 4.9.1 |
| Source | NVIDIA/nccl `v2.30.7-1` plus this fork's commits (`git describe` in `build/<cuda>/<describe>/`) |
| Measured | 2026-10-11, one run each, nothing else building. Single runs: treat differences under about 10% as noise |

## Results

307 units are expected from a `make -n` dry run; 308 are timed (one extra recipe line with `-o`).

| Run | Wall | Summed unit time | ccache hits | Data |
|---|---|---|---|---|
| Cold: empty cache | 109 s | 1,963 s | 0 / 128 (nvcc only; host C++ not yet cached) | [`cold.tsv`](build_timings/cold.tsv) |
| Same commit, tree cleared, cache warm | 31 s | 314 s | 266 / 266 | [`warm-same-commit.tsv`](build_timings/warm-same-commit.tsv) |
| New commit, tree cleared, cache warm | 29 s | 342 s | 265 / 266 | [`warm-new-commit.tsv`](build_timings/warm-new-commit.tsv) |
| New commit, but built in a directory named after the commit (the bug, see below) | 112 s | 2,011 s | 14 / 266 | not kept |

"Summed unit time" adds up every unit's own duration; dividing by wall time gives the average
parallelism (about 18 of 20 jobs when cold).

### Where the time goes

| Kind | Units | Cold | Warm (new commit) |
|---|---|---|---|
| `nvcc` (device kernels) | 166 | 1,732 s, slowest 37.9 s | 341 s, slowest 18.3 s |
| `cxx` (host C++) | 138 | 230 s, slowest 4.9 s | 0 s (all cached) |
| `link` | 4 | 0.3 s | 0.3 s |

Cold, the slowest units are the `all_reduce_*` kernels at 30 to 38 s each. Warm, 34 of the 35
units over 5 s are `symmetric/…` kernels (`reduce_scatter_gin_*`, up to 18 s): they are never
cached, so they set the floor on a warm build.

## What the cache can and cannot do

- **nvcc is wrapped directly** (`NVCC="ccache nvcc"`). **Host C++ is routed through ccache by
  `tuned/timed-shell.sh`**, because NCCL passes `-ccbin $(CXX)` to nvcc and `CXX` must stay a bare
  compiler. Only recipes that do not run nvcc get ccache first on `PATH`.
- **Uncacheable: the `-dw` kernels.** ccache's own breakdown of the uncacheable calls on a warm run:
  38 "called for linking" (the symmetric kernels compile with `nvcc -dw`, which ccache does not
  recognise as a compile), 20 "no input file" (`nvcc --version` probes) and 15 "unsupported compiler
  option". Caching them would need a different compile mode, which could change the output, so it
  is not done.
- **ccache keys on paths.** The first attempt built inside `build/<cuda>/<describe>/nccl`; a new
  commit changes `<describe>`, so every `-o`, `-I` and generated-source path changed and a new
  commit hit only 14 of 266 (112 s, same as cold). The fix is a fixed work path,
  `build/<cuda>/work/nccl`, with `lib/` and `include/` copied into the `<describe>` tree after the
  checks pass.
- **The git hash is stripped from every C++ command line except `git_version.cc`.** NCCL adds
  `-DNCCL_GIT_COMMIT_HASH` and `-DNCCL_GIT_BRANCH` to all of them, and only `git_version.cc` reads
  them. That one file is the single miss on a new commit.
- **NCCL's make does not track the variables we pass** (`NVCC_GENCODE`, `NCCL_SUFFIX`, the git
  hash). `tuned/build.sh` records them, plus a hash of uncommitted changes, in `.params` and starts
  from an empty tree when they differ; otherwise a rebuild silently reuses a stale `nccl.h`
  (this happened once: the `-tuned` suffix did not appear until the stamp was added).

## Reproduce

```bash
# Cold: clear this repo's ccache first (it only holds NCCL objects).
CCACHE_DIR=build/cu134/ccache ccache -C
CUDA_TAG=cu134 bash tuned/build.sh gb10          # cold run; prints [done/total] and a summary

# Warm, same commit: remove only the work tree, then build again.
rm -rf build/cu134/work/nccl
CUDA_TAG=cu134 bash tuned/build.sh gb10

# Warm, new commit: commit anything, then build again.
CUDA_TAG=cu134 bash tuned/build.sh gb10
```

`build/<cuda>/<describe>/timing.tsv` holds one line per unit: seconds, kind, output, exit status.
`NCCL_TUNED_NO_CCACHE=1` builds without the cache. The summary and the slowest-10 list come from
`tuned/build.sh`; `tuned/timed-shell.sh` produces the per-unit lines.

## Not covered

- One machine, one toolkit, one run per row: no variance estimate.
- Only NCCL. PyTorch (ninja) and llama.cpp are not timed this way.
- The effect of `-j` other than 20, and of ccache sizes other than the 20 GB cap, was not tried.
