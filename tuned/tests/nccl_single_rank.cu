// tuned/tests/nccl_single_rank.cu -- functional test of a tuned NCCL build on ONE GPU (one rank).
//
// This node has a single GPU, so only single-rank communicators can be created (NCCL refuses two ranks
// on one device). With one rank NCCL takes its one-rank path (src/device/onerank.cu: a plain device copy,
// or a PreMulSum kernel), so every collective is the identity (or a scale, for PreMulSum).
//
// What this proves: the library matches its header; a communicator initialises on the GPU this build targets
// (sm_121a); every data type x reduction x size is accepted and returns the right bytes through that path;
// the PreMulSum kernel runs; the argument checks behave. What it does NOT prove: any multi-rank algorithm or
// transport (ring/tree/NVLS, shared memory, network) and the multi-rank device kernels, including the FP8
// multicast (LDMC) kernels whose build gate this fork patches (NVIDIA/nccl#2420). The patch is checked at build
// time by gpu_tuned_verify_arch (single-arch), not here. docs/ARCHITECTURE.md G3: this narrower gate is a
// recorded decision, not a default.
//
// ncclAvg on FP8 is REJECTED with ncclInvalidArgument: NCCL implements it under a host-side
// __CUDA_FP8_TYPES_EXIST__ guard that is not defined when enqueue.cc is compiled. NVIDIA's own
// libnccl2 2.31.2+cuda13.3 package behaves the same (measured 2026-10-18 with this test), so the test expects
// the rejection and fails if a future NCCL starts accepting it, so that the expectation gets reviewed.
//
// Usage: nccl_single_rank   (exit 0 and "PASSED <n> checks" on success)
#include <cuda_runtime.h>
#include <nccl.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(2); } } while (0)

static int g_checks = 0, g_failed = 0;
static ncclComm_t g_comm;
static cudaStream_t g_stream;

static void fail(const char *what, const char *detail) {
  ++g_failed;
  std::fprintf(stderr, "FAIL: %s: %s\n", what, detail);
}

static bool nccl_ok(ncclResult_t r, const char *what) {
  if (r == ncclSuccess) return true;
  char buf[256];
  std::snprintf(buf, sizeof buf, "%s -> %s (%s)", what, ncclGetErrorString(r), ncclGetLastError(g_comm));
  fail(what, buf);
  return false;
}

struct Dtype { const char *name; ncclDataType_t t; size_t size; };
static const Dtype kDtypes[] = {
  {"int8", ncclInt8, 1},       {"uint8", ncclUint8, 1},      {"int32", ncclInt32, 4},
  {"uint32", ncclUint32, 4},   {"int64", ncclInt64, 8},      {"uint64", ncclUint64, 8},
  {"half", ncclFloat16, 2},    {"float", ncclFloat32, 4},    {"double", ncclFloat64, 8},
  {"bfloat16", ncclBfloat16, 2},
#if defined(NCCL_VERSION_CODE) && NCCL_VERSION_CODE >= NCCL_VERSION(2, 24, 0)
  {"fp8_e4m3", ncclFloat8e4m3, 1}, {"fp8_e5m2", ncclFloat8e5m2, 1},
#endif
};
struct Op { const char *name; ncclRedOp_t op; };
static const Op kOps[] = {{"sum", ncclSum}, {"prod", ncclProd}, {"max", ncclMax}, {"min", ncclMin}, {"avg", ncclAvg}};

// Finite values for every type: all bytes below 0x60 (no all-ones exponent, so no Inf/NaN in fp16/bf16/fp8/fp32/fp64).
static void fill(std::vector<uint8_t> &v) {
  for (size_t i = 0; i < v.size(); ++i) v[i] = (uint8_t)((i * 7 + 3) % 0x60);
}

static void *dev_in, *dev_out;

static bool run_and_compare(const char *what, const std::vector<uint8_t> &want, size_t bytes) {
  CUDA(cudaStreamSynchronize(g_stream));
  std::vector<uint8_t> got(bytes);
  CUDA(cudaMemcpy(got.data(), dev_out, bytes, cudaMemcpyDeviceToHost));
  ++g_checks;
  if (std::memcmp(got.data(), want.data(), bytes) != 0) {
    size_t i = 0; while (i < bytes && got[i] == want[i]) ++i;
    char buf[160]; std::snprintf(buf, sizeof buf, "first differing byte %zu (got 0x%02x want 0x%02x) of %zu", i, got[i], want[i], bytes);
    fail(what, buf);
    return false;
  }
  return true;
}

static void stage(const std::vector<uint8_t> &in) {
  CUDA(cudaMemcpy(dev_in, in.data(), in.size(), cudaMemcpyHostToDevice));
  CUDA(cudaMemset(dev_out, 0xAA, in.size()));
}

int main() {
  int version = 0;
  if (ncclGetVersion(&version) != ncclSuccess || version != NCCL_VERSION_CODE) {
    std::fprintf(stderr, "FAIL: library version code %d does not match the header's %d (wrong libnccl for this nccl.h)\n", version, NCCL_VERSION_CODE);
    return 1;
  }
  std::printf("NCCL library %d.%d.%d matches its header\n", version / 10000, (version / 100) % 100, version % 100);
  int dev = 0;
  CUDA(cudaGetDevice(&dev));
  cudaDeviceProp prop; CUDA(cudaGetDeviceProperties(&prop, dev));
  std::printf("device %d: %s (sm_%d%d)\n", dev, prop.name, prop.major, prop.minor);
  CUDA(cudaStreamCreate(&g_stream));
  if (ncclCommInitAll(&g_comm, 1, &dev) != ncclSuccess) { std::fprintf(stderr, "FAIL: ncclCommInitAll\n"); return 1; }

  const size_t counts[] = {1, 1000, 1u << 20};
  const size_t max_bytes = (size_t)(1u << 20) * 8;
  CUDA(cudaMalloc(&dev_in, max_bytes));
  CUDA(cudaMalloc(&dev_out, max_bytes));

  // AllReduce: every data type x reduction x size is the identity on one rank.
  for (const Dtype &d : kDtypes) {
    for (const Op &o : kOps) {
      for (size_t n : counts) {
        size_t bytes = n * d.size;
        std::vector<uint8_t> in(bytes); fill(in);
        stage(in);
        char what[96]; std::snprintf(what, sizeof what, "AllReduce %s %s n=%zu", d.name, o.name, n);
        bool fp8 = d.size == 1 && d.name[0] == 'f';
        if (fp8 && o.op == ncclAvg) {  // expected rejection, see the header comment
          ncclResult_t r = ncclAllReduce(dev_in, dev_out, n, d.t, o.op, g_comm, g_stream);
          ++g_checks;
          if (r != ncclInvalidArgument) {
            char buf[160]; std::snprintf(buf, sizeof buf, "expected ncclInvalidArgument for FP8 average, got %s: review the expectation", ncclGetErrorString(r));
            fail(what, buf);
          }
          continue;
        }
        if (!nccl_ok(ncclAllReduce(dev_in, dev_out, n, d.t, o.op, g_comm, g_stream), what)) continue;
        run_and_compare(what, in, bytes);
      }
    }
  }

  // The other collectives, float32, a small and a large size.
  const size_t fcounts[] = {1, 4096, 1u << 20};
  for (size_t n : fcounts) {
    size_t bytes = n * 4;
    std::vector<uint8_t> in(bytes); fill(in);
    char what[96];
    stage(in); std::snprintf(what, sizeof what, "Broadcast float n=%zu", n);
    if (nccl_ok(ncclBroadcast(dev_in, dev_out, n, ncclFloat32, 0, g_comm, g_stream), what)) run_and_compare(what, in, bytes);
    stage(in); std::snprintf(what, sizeof what, "Reduce float n=%zu", n);
    if (nccl_ok(ncclReduce(dev_in, dev_out, n, ncclFloat32, ncclSum, 0, g_comm, g_stream), what)) run_and_compare(what, in, bytes);
    stage(in); std::snprintf(what, sizeof what, "AllGather float n=%zu", n);
    if (nccl_ok(ncclAllGather(dev_in, dev_out, n, ncclFloat32, g_comm, g_stream), what)) run_and_compare(what, in, bytes);
    stage(in); std::snprintf(what, sizeof what, "ReduceScatter float n=%zu", n);
    if (nccl_ok(ncclReduceScatter(dev_in, dev_out, n, ncclFloat32, ncclSum, g_comm, g_stream), what)) run_and_compare(what, in, bytes);
    // Send/Recv to oneself must be grouped, or the send waits for a receive that was never posted.
    stage(in); std::snprintf(what, sizeof what, "Send/Recv to self float n=%zu", n);
    bool ok = nccl_ok(ncclGroupStart(), what);
    ok = ok && nccl_ok(ncclSend(dev_in, n, ncclFloat32, 0, g_comm, g_stream), what);
    ok = ok && nccl_ok(ncclRecv(dev_out, n, ncclFloat32, 0, g_comm, g_stream), what);
    ok = nccl_ok(ncclGroupEnd(), what) && ok;
    if (ok) run_and_compare(what, in, bytes);
  }

  // PreMulSum with a host scalar: out = 2 * in for float32.
  {
    const size_t n = 4096, bytes = n * 4;
    std::vector<float> in(n), want(n);
    for (size_t i = 0; i < n; ++i) { in[i] = (float)(i % 251) * 0.5f; want[i] = in[i] * 2.0f; }
    CUDA(cudaMemcpy(dev_in, in.data(), bytes, cudaMemcpyHostToDevice));
    CUDA(cudaMemset(dev_out, 0xAA, bytes));
    float scalar = 2.0f; ncclRedOp_t premul;
    if (nccl_ok(ncclRedOpCreatePreMulSum(&premul, &scalar, ncclFloat32, ncclScalarHostImmediate, g_comm), "RedOpCreatePreMulSum")) {
      if (nccl_ok(ncclAllReduce(dev_in, dev_out, n, ncclFloat32, premul, g_comm, g_stream), "AllReduce PreMulSum float")) {
        std::vector<uint8_t> w(bytes); std::memcpy(w.data(), want.data(), bytes);
        run_and_compare("AllReduce PreMulSum float scale 2", w, bytes);
      }
      ncclRedOpDestroy(premul, g_comm);
    }
  }

  CUDA(cudaFree(dev_in)); CUDA(cudaFree(dev_out));
  ncclCommDestroy(g_comm);
  CUDA(cudaStreamDestroy(g_stream));
  if (g_failed) { std::printf("FAILED %d of %d checks\n", g_failed, g_checks); return 1; }
  std::printf("PASSED %d checks\n", g_checks);
  return 0;
}
