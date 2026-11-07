# NCCL 2.32.3 for RTX 50-series (Blackwell consumer, desktop/laptop, SM 120a) (cu134) — v0.20261107.0-rtx50-cu134

Static archive and headers of NCCL 2.32.3, built single-arch (sm_120a) with CUDA 13.4, for
PyTorch's USE_SYSTEM_NCCL=1 USE_STATIC_NCCL=1 and for llama.cpp's static NCCL. No shared library is shipped.

## Version
The library reports itself as `NCCL version 2.32.3-rtx50-cu134.20261107.0` (also in `nccl.h` as `NCCL_SUFFIX "-rtx50-cu134.20261107.0"`):
NCCL 2.32.3, built for variant `rtx50` with toolkit `cu134`, release label `20261107.0` (the same label as the tag
`v0.20261107.0-rtx50-cu134`). Upstream NCCL appends `+cuda<major>.<minor>` to this line; this fork drops it (one line in src/init.cc), because the
suffix already names the toolkit. The numeric version code and the library file names are unchanged.

## Source
- Base: NVIDIA/nccl tag `v2.32.3-1` (the NCCL release NVIDIA ships for CUDA 13.4), plus this fork's changes: the
  single-arch fix in src/device/Makefile (NVIDIA/nccl issue 2420, still open upstream), the one-line version-line change in src/init.cc
  (drops the `+cuda` tail) and the `tuned/` build tooling.
- Fork commit: `f2b26b2d0f26b28eddc2e310bc7fcbec9bd135f3` (`v2.32.3-1-34-gf2b26b2d`).

## Tested
tuned/tests/nccl_single_rank.cu on one RTX 5090 Laptop GPU (sm_120): 196 checks against the static archive and 196 against the shared build,
all passed. **Single rank only:** no multi-rank algorithm or transport (NVLink, network, shared memory) was exercised, and neither
were the multi-rank device kernels. FP8 average is rejected with an invalid-argument error, as in NVIDIA's own 2.31.2 package.

## Files
`nccl-tuned-…tar.gz` with `lib/libnccl_static.a`, `include/`, `LICENSE.txt`, `BUILD_INFO`, `BUILD_INFO.json`; a `.sha256` sidecar; the test and smoke logs.
