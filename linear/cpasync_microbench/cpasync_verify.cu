/**
 * 微基准：在 Ada/Ampere（含 RTX 4070 / sm_89）上验证非 bulk cp.async（4B）+
 * commit_group / wait_group(0) 的语义，并与同步路径对照计时。
 *
 * 编译需 -std=c++17，架构建议 sm_89（可通过 CMake 变量覆盖）。
 */
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check_cuda(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "[ERR] %s: %s\n", what, cudaGetErrorString(e));
    std::exit(1);
  }
}

__device__ __forceinline__ void cp_async_4b(void* smem_dst, const void* gmem_src) {
  unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(smem_addr), "l"(gmem_src)
               : "memory");
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

/** 每线程发起一条 4B cp.async：src[t] -> smem[t]，再 commit / wait，最后写回 dst。 */
__global__ void kernel_cpasync(const float* __restrict__ src, float* __restrict__ dst, int n) {
  extern __shared__ __align__(16) char smem_bytes[];
  auto* smem = reinterpret_cast<float*>(smem_bytes);
  const int t = static_cast<int>(threadIdx.x);
  if (t < n) {
    cp_async_4b(smem + t, src + t);
  }
  cp_async_commit();
  cp_async_wait_all();
  __syncthreads();
  if (t < n) {
    dst[t] = smem[t];
  }
}

/** 对照：标量经寄存器写入 smem（无 cp.async）。 */
__global__ void kernel_sync(const float* __restrict__ src, float* __restrict__ dst, int n) {
  extern __shared__ __align__(16) char smem_bytes[];
  auto* smem = reinterpret_cast<float*>(smem_bytes);
  const int t = static_cast<int>(threadIdx.x);
  if (t < n) {
    smem[t] = src[t];
  }
  __syncthreads();
  if (t < n) {
    dst[t] = smem[t];
  }
}

static void usage(const char* argv0) {
  std::fprintf(stderr,
               "Usage: %s [--iters N] [--warmup W] [--n ELEM]\n"
               "  Defaults: iters=1000 warmup=50 n=256 (single block, 256 threads)\n",
               argv0);
}

int main(int argc, char** argv) {
  int iters = 1000;
  int warmup = 50;
  int n = 256;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
      iters = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      warmup = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
      n = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
      usage(argv[0]);
      return 0;
    } else {
      usage(argv[0]);
      return 1;
    }
  }
  if (n <= 0 || n > 1024) {
    std::fprintf(stderr, "n must be in 1..1024 (single block)\n");
    return 1;
  }

  check_cuda(cudaSetDevice(0), "cudaSetDevice");

  std::printf("[cpasync_microbench] n=%d threads=%d bytes_moved=%d\n", n, n, n * 4);

  std::vector<float> h_src(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    h_src[static_cast<size_t>(i)] = static_cast<float>(i) * 0.25f + 1.0f;
  }

  float *d_src = nullptr, *d_dst_async = nullptr, *d_dst_sync = nullptr;
  check_cuda(cudaMalloc(&d_src, static_cast<size_t>(n) * sizeof(float)), "cudaMalloc src");
  check_cuda(cudaMalloc(&d_dst_async, static_cast<size_t>(n) * sizeof(float)), "cudaMalloc dst_async");
  check_cuda(cudaMalloc(&d_dst_sync, static_cast<size_t>(n) * sizeof(float)), "cudaMalloc dst_sync");
  check_cuda(cudaMemcpy(d_src, h_src.data(), static_cast<size_t>(n) * sizeof(float), cudaMemcpyHostToDevice),
              "H2D");

  const int threads = 256;
  const size_t smem_sz = static_cast<size_t>(n) * sizeof(float);

  kernel_cpasync<<<1, threads, smem_sz>>>(d_src, d_dst_async, n);
  check_cuda(cudaGetLastError(), "launch kernel_cpasync");
  check_cuda(cudaDeviceSynchronize(), "sync after cpasync");

  kernel_sync<<<1, threads, smem_sz>>>(d_src, d_dst_sync, n);
  check_cuda(cudaGetLastError(), "launch kernel_sync");
  check_cuda(cudaDeviceSynchronize(), "sync after sync-kernel");

  std::vector<float> h_async(static_cast<size_t>(n)), h_sync(static_cast<size_t>(n));
  check_cuda(cudaMemcpy(h_async.data(), d_dst_async, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost),
              "D2H async");
  check_cuda(cudaMemcpy(h_sync.data(), d_dst_sync, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost),
              "D2H sync");

  int bad = 0;
  for (int i = 0; i < n; ++i) {
    const float a = h_async[static_cast<size_t>(i)];
    const float r = h_src[static_cast<size_t>(i)];
    if (a != r) {
      if (bad < 4) {
        std::printf("[FAIL] i=%d cp.async got %g expect %g\n", i, static_cast<double>(a), static_cast<double>(r));
      }
      ++bad;
    }
  }
  if (bad != 0) {
    std::printf("[FAIL] cp.async mismatch count %d / %d\n", bad, n);
    return 1;
  }
  for (int i = 0; i < n; ++i) {
    if (h_sync[static_cast<size_t>(i)] != h_src[static_cast<size_t>(i)]) {
      std::printf("[FAIL] sync kernel wrong at i=%d\n", i);
      return 1;
    }
  }
  std::printf("[OK] cp.async path matches host pattern; sync path matches.\n");

  cudaEvent_t e0{}, e1{};
  check_cuda(cudaEventCreate(&e0), "event0");
  check_cuda(cudaEventCreate(&e1), "event1");

  for (int w = 0; w < warmup; ++w) {
    kernel_cpasync<<<1, threads, smem_sz>>>(d_src, d_dst_async, n);
  }
  check_cuda(cudaDeviceSynchronize(), "warmup sync");

  check_cuda(cudaEventRecord(e0), "record0");
  for (int i = 0; i < iters; ++i) {
    kernel_cpasync<<<1, threads, smem_sz>>>(d_src, d_dst_async, n);
  }
  check_cuda(cudaEventRecord(e1), "record1");
  check_cuda(cudaEventSynchronize(e1), "sync e1");
  float ms_async = 0.f;
  check_cuda(cudaEventElapsedTime(&ms_async, e0, e1), "elapsed async");

  for (int w = 0; w < warmup; ++w) {
    kernel_sync<<<1, threads, smem_sz>>>(d_src, d_dst_sync, n);
  }
  check_cuda(cudaDeviceSynchronize(), "warmup2");
  check_cuda(cudaEventRecord(e0), "record0b");
  for (int i = 0; i < iters; ++i) {
    kernel_sync<<<1, threads, smem_sz>>>(d_src, d_dst_sync, n);
  }
  check_cuda(cudaEventRecord(e1), "record1b");
  check_cuda(cudaEventSynchronize(e1), "sync e1b");
  float ms_sync = 0.f;
  check_cuda(cudaEventElapsedTime(&ms_sync, e0, e1), "elapsed sync");

  const double bytes = static_cast<double>(n) * 4.0 * static_cast<double>(iters);
  std::printf("[TIME] cp.async kernel: total %.3f ms  (%.3f us/iter)  ~%.2f GB/s effective (R+W smem)\n",
              static_cast<double>(ms_async), static_cast<double>(ms_async) * 1000.0 / static_cast<double>(iters),
              (bytes / 1e9) / (static_cast<double>(ms_async) / 1000.0));
  std::printf("[TIME] sync  kernel: total %.3f ms  (%.3f us/iter)\n",
              static_cast<double>(ms_sync), static_cast<double>(ms_sync) * 1000.0 / static_cast<double>(iters));

  cudaFree(d_src);
  cudaFree(d_dst_async);
  cudaFree(d_dst_sync);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return 0;
}
