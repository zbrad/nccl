# NCCL 2.32.3 for RTX 4080/4090 (Ada Lovelace, SM 89) (cu134) — v0.20261107.0-rtx40-cu134

Static archive and headers of NCCL 2.32.3, built single-arch (sm_89) with CUDA 13.4, for
PyTorch's USE_SYSTEM_NCCL=1 USE_STATIC_NCCL=1 and for llama.cpp's static NCCL. No shared library is shipped.

## Version
The library reports itself as `NCCL version 2.32.3-rtx40-cu134.20261107.0` (also in `nccl.h` as `NCCL_SUFFIX "-rtx40-cu134.20261107.0"`):
NCCL 2.32.3, built for variant `rtx40` with toolkit `cu134`, release label `20261107.0` (the same label as the tag
`v0.20261107.0-rtx40-cu134`). Upstream NCCL appends `+cuda<major>.<minor>` to this line; this fork drops it (one line in src/init.cc), because the
suffix already names the toolkit. The numeric version code and the library file names are unchanged.

## Source
- Base: NVIDIA/nccl tag `v2.32.3-1` (the NCCL release NVIDIA ships for CUDA 13.4), plus this fork's changes: the
  single-arch fix in src/device/Makefile (NVIDIA/nccl issue 2420, still open upstream), the one-line version-line change in src/init.cc
  (drops the `+cuda` tail) and the `tuned/` build tooling.
- Fork commit: `f2b26b2d0f26b28eddc2e310bc7fcbec9bd135f3` (`v2.32.3-1-34-gf2b26b2d`).

## Tested
tuned/tests/nccl_single_rank.cu on one RTX 4090 (sm_89): 196 checks against the static archive and 196 against the shared build,
all passed. **Single rank only:** no multi-rank algorithm or transport (NVLink, network, shared memory) was exercised, and neither
were the multi-rank device kernels. FP8 average is rejected with an invalid-argument error, as in NVIDIA's own 2.31.2 package.

**No FP8 reductions on this GPU.** NCCL refuses every FP8 (e4m3, e5m2) reducing collective on devices below sm_90
(`src/enqueue/enqueue.cc`: "FP8 reduction support begins with sm90 capable devices"), so on sm_89 FP8 AllReduce
returns an invalid-argument error for every op; the test asserts that refusal. That check exempts AllGather,
Broadcast, AlltoAll, Scatter and Gather (read from the source; FP8 is not exercised for those collectives here).

## Files
`nccl-tuned-…tar.gz` with `lib/libnccl_static.a`, `include/`, `LICENSE.txt`, `BUILD_INFO`, `BUILD_INFO.json`; a `.sha256` sidecar; the test and smoke logs.
