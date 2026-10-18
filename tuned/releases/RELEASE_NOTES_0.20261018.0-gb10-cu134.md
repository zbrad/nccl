**DRAFT — not yet published.** Review, correct and remove this line, then commit the file.

# NCCL 2.32.3 for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated) (cu134) — v0.20261018.0-gb10-cu134

Static archive and headers of NCCL 2.32.3, built single-arch (sm_121a) with CUDA 13.4, for
PyTorch's USE_SYSTEM_NCCL=1 USE_STATIC_NCCL=1 and for llama.cpp's static NCCL. No shared library is shipped.

## Version
The library reports itself as `NCCL version 2.32.3-gb10-cu134.20261018.0+cuda13.4` (also in `nccl.h` as `NCCL_SUFFIX "-gb10-cu134.20261018.0"`):
NCCL 2.32.3, built for variant `gb10` with toolkit `cu134`, release label `20261018.0` (the same label as the tag
`v0.20261018.0-gb10-cu134`); `+cuda13.4` is NCCL's own record of the CUDA it was compiled with. The numeric version code and the
library file names are unchanged.

## Source
- Base: NVIDIA/nccl tag `v2.32.3-1` (the NCCL release NVIDIA ships for CUDA 13.4), plus this fork's changes: the
  single-arch fix in src/device/Makefile (NVIDIA/nccl issue 2420, still open upstream) and the `tuned/` build tooling.
- Fork commit: `eca85ee341dbd284fd813ff39193084980f8d5da` (`v2.32.3-1-24-geca85ee3`).

## Tested
tuned/tests/nccl_single_rank.cu on one GB10 GPU: 196 checks against the static archive and 196 against the shared build,
all passed. **Single rank only:** no multi-rank algorithm or transport (NVLink, network, shared memory) was exercised, and neither
were the multi-rank device kernels. FP8 average is rejected with an invalid-argument error, as in NVIDIA's own 2.31.2 package.

## Files
`nccl-tuned-…tar.gz` with `lib/libnccl_static.a`, `include/`, `LICENSE.txt`, `BUILD_INFO`, `BUILD_INFO.json`; a `.sha256` sidecar; the test and smoke logs.
