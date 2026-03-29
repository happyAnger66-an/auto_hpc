/*
 * Tiled linear Y = X @ Wt + b（FP32），与 cutedsl/benchmark_linear.py 的 pad 与 tile 约定一致：
 *   X [M,K]、Wt [K,N] 行主序，BM=BN=64、BK=16，双槽 smem；预取路径使用 16B cp.async（float4）。
 *
 * 需 sm_80+（Ada 4070 / sm_89 可用）。对齐：K、N 为 pad 后尺寸，k_tile % 16 == 0，bn % 64 == 0，
 * 每线程负责 flat 偏移 4*tidx 起的连续 4 个 float，全局地址 16B 对齐。
 */
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdlib>

namespace {

constexpr int kBM = 64;
constexpr int kBN = 64;
constexpr int kBK = 16;
constexpr int kThreads = 256;
constexpr int kTileFloats = kBM * kBK;  // 1024
constexpr int kSmemFloats = 4 * kTileFloats;

__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src) {
  unsigned sa = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(sa), "l"(gmem_src) : "memory");
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

__device__ void load_A_tile_cpasync(const float* x, int bm, int K, int k0, int buf, float* pool) {
  float* dst_base = pool + buf * kTileFloats;
  const int t = static_cast<int>(threadIdx.x);
  const int o = t * 4;
  const int r = o >> 4;
  const int c = o & 15;
  const float* g = x + (bm + r) * K + k0 + c;
  float* s = dst_base + r * kBK + c;
#if defined(CPASYNC_LINEAR_USE_SYNC_LOAD)
  *reinterpret_cast<float4*>(s) = *reinterpret_cast<const float4*>(g);
#else
  cp_async_16(s, g);
#endif
}

__device__ void load_B_tile_cpasync(const float* wt, int bn, int N, int k0, int buf, float* pool) {
  float* dst_base = pool + 2 * kTileFloats + buf * kTileFloats;
  const int t = static_cast<int>(threadIdx.x);
  const int o = t * 4;
  const int r = o >> 6;
  const int c = o & 63;
  const float* g = wt + (k0 + r) * N + bn + c;
  float* s = dst_base + r * kBN + c;
#if defined(CPASYNC_LINEAR_USE_SYNC_LOAD)
  *reinterpret_cast<float4*>(s) = *reinterpret_cast<const float4*>(g);
#else
  cp_async_16(s, g);
#endif
}

__global__ void linear_cpasync_tiled_kernel(float* __restrict__ y, const float* __restrict__ x,
                                            const float* __restrict__ wt, const float* __restrict__ b, int M,
                                            int N, int K) {
  extern __shared__ __align__(16) float smem_pool[];

  const int bidx = blockIdx.x;
  const int bidy = blockIdx.y;
  const int bm = bidx * kBM;
  const int bn = bidy * kBN;
  const int tidx = static_cast<int>(threadIdx.x);
  const int ti = tidx >> 4;
  const int tj = tidx & 15;

  float frag[4][4];
#pragma unroll
  for (int ii = 0; ii < 4; ++ii)
#pragma unroll
    for (int jj = 0; jj < 4; ++jj)
      frag[ii][jj] = 0.f;

  // Prologue：tile 0 → buffer 0（async + wait，与循环内预取一致）
  load_A_tile_cpasync(x, bm, K, 0, 0, smem_pool);
  load_B_tile_cpasync(wt, bn, N, 0, 0, smem_pool);
#if !defined(CPASYNC_LINEAR_USE_SYNC_LOAD)
  cp_async_commit();
  cp_async_wait_all();
#endif
  __syncthreads();

  for (int k_tile = 0; k_tile < K; k_tile += kBK) {
    const int kn = k_tile / kBK;
    // 预取下一条 K 到 wp（与当前计算用的 rp 槽位不同，可与 MMA 重叠）
    if (k_tile + kBK < K) {
      const int kb_n = kn + 1;
      const int wp = 1 - (kn & 1);
      const int k_next = kb_n * kBK;
      load_A_tile_cpasync(x, bm, K, k_next, wp, smem_pool);
      load_B_tile_cpasync(wt, bn, N, k_next, wp, smem_pool);
#if !defined(CPASYNC_LINEAR_USE_SYNC_LOAD)
      cp_async_commit();
#else
      __syncthreads();
#endif
    }

    const int rp = kn & 1;
    float* sAp = smem_pool + rp * kTileFloats;
    float* sBp = smem_pool + 2 * kTileFloats + rp * kTileFloats;

#pragma unroll
    for (int kk = 0; kk < kBK; ++kk) {
#pragma unroll
      for (int ii = 0; ii < 4; ++ii) {
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
          frag[ii][jj] += sAp[(ti * 4 + ii) * kBK + kk] * sBp[kk * kBN + (tj * 4 + jj)];
        }
      }
    }

#if !defined(CPASYNC_LINEAR_USE_SYNC_LOAD)
    __syncthreads();
    if (k_tile + kBK < K) {
      cp_async_wait_all();
      __syncthreads();
    }
#else
    __syncthreads();
#endif
  }

#pragma unroll
  for (int ii = 0; ii < 4; ++ii) {
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const int gi = bm + ti * 4 + ii;
      const int gj = bn + tj * 4 + jj;
      y[gi * N + gj] = frag[ii][jj] + b[gj];
    }
  }
}

void launch_linear_cpasync(const torch::Tensor& y, const torch::Tensor& x, const torch::Tensor& wt,
                           const torch::Tensor& b) {
  TORCH_CHECK(x.is_cuda() && wt.is_cuda() && b.is_cuda() && y.is_cuda(), "tensors must be CUDA");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "X must be float32");
  TORCH_CHECK(wt.scalar_type() == torch::kFloat32 && b.scalar_type() == torch::kFloat32, "Wt,b float32");
  TORCH_CHECK(x.is_contiguous() && wt.is_contiguous() && b.is_contiguous() && y.is_contiguous(),
              "tensors must be contiguous");
  const int M = static_cast<int>(x.size(0));
  const int K = static_cast<int>(x.size(1));
  const int K2 = static_cast<int>(wt.size(0));
  const int N = static_cast<int>(wt.size(1));
  TORCH_CHECK(K == K2, "K mismatch X vs Wt");
  TORCH_CHECK(y.size(0) == M && y.size(1) == N, "Y shape");
  TORCH_CHECK(b.size(0) == N, "bias shape");
  TORCH_CHECK(M % kBM == 0 && N % kBN == 0 && K % kBK == 0, "M,N,K must be multiples of 64,64,16");

  const dim3 grid((M + kBM - 1) / kBM, (N + kBN - 1) / kBN);
  const dim3 block(kThreads);
  const size_t smem_bytes = kSmemFloats * sizeof(float);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  linear_cpasync_tiled_kernel<<<grid, block, smem_bytes, stream>>>(
      y.data_ptr<float>(), x.data_ptr<float>(), wt.data_ptr<float>(), b.data_ptr<float>(), M, N, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor cpasync_linear_forward(torch::Tensor x, torch::Tensor wt, torch::Tensor b) {
  auto y = torch::empty({x.size(0), wt.size(1)}, x.options());
  launch_linear_cpasync(y, x, wt, b);
  return y;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &cpasync_linear_forward, "Y = X @ Wt + b (cp.async prefetch, float32)");
}
