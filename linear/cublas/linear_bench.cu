/*
 * FP32 Linear: Y = X @ W^T + b
 *
 * 与 PyTorch nn.Linear 一致：X/W 在主机为行主序 [M,K]、[N,K]，b 为 [N]。
 *
 * 设备端按 cuBLAS 惯例使用列主序：
 *   - cublasSetMatrix 将主机行主序拷为设备列主序
 *   - A: M×K 列主序 lda=M；B: W^T 为 K×N 列主序 ldb=K；C: M×N 列主序 ldc=M
 *   - cublasSgemm(handle, OP_N, OP_N, M, N, K, ...) 即 C = A * B
 * bias: 预生成列主序 M×N 展开向量（每列填 b[n]），cublasSaxpy 一次完成 Y += broadcast(b)
 *
 * 用法: ./linear_bench M N K warmup iters
 * 输出: OK ms_per_iter <float> gflops <float> method cublas_sgemm_colmajor
 *
 * GFLOPS = (2*M*N*K + M*N) / time_s / 1e9
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

static void checkCuda(cudaError_t e, const char *msg) {
  if (e != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
    std::exit(1);
  }
}

static void checkCublas(cublasStatus_t s, const char *msg) {
  if (s != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "%s: cublas error %d\n", msg, (int)s);
    std::exit(1);
  }
}

int main(int argc, char **argv) {
  if (argc != 6) {
    fprintf(stderr, "Usage: %s M N K warmup iters\n", argv[0]);
    return 1;
  }
  const int M = std::atoi(argv[1]);
  const int N = std::atoi(argv[2]);
  const int K = std::atoi(argv[3]);
  const int warmup = std::atoi(argv[4]);
  const int iters = std::atoi(argv[5]);
  if (M <= 0 || N <= 0 || K <= 0 || warmup < 0 || iters < 1) {
    fprintf(stderr, "Invalid M,N,K,warmup,iters\n");
    return 1;
  }

  const size_t elems_x = static_cast<size_t>(M) * static_cast<size_t>(K);
  const size_t elems_w = static_cast<size_t>(N) * static_cast<size_t>(K);
  const size_t elems_y = static_cast<size_t>(M) * static_cast<size_t>(N);
  const size_t elems_b = static_cast<size_t>(N);

  std::vector<float> h_x(elems_x);
  std::vector<float> h_w(elems_w);
  std::vector<float> h_b(elems_b);
  for (size_t i = 0; i < elems_x; ++i) {
    h_x[i] = static_cast<float>(static_cast<int>(i % 911) * 1e-3f);
  }
  for (size_t i = 0; i < elems_w; ++i) {
    h_w[i] = static_cast<float>(static_cast<int>(i % 733) * 1e-3f);
  }
  for (size_t i = 0; i < elems_b; ++i) {
    h_b[i] = static_cast<float>(static_cast<int>(i % 401) * 1e-3f);
  }

  // W^T as K×N column-major (same storage as row-major K×N with lda=K)
  std::vector<float> h_wt(static_cast<size_t>(K) * static_cast<size_t>(N));
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      h_wt[static_cast<size_t>(k) + static_cast<size_t>(n) * static_cast<size_t>(K)] =
          h_w[static_cast<size_t>(n) * static_cast<size_t>(K) + static_cast<size_t>(k)];
    }
  }

  // bias broadcast: M×N column-major, column n filled with b[n]
  std::vector<float> h_bias_bc(elems_y);
  for (int n = 0; n < N; ++n) {
    for (int m = 0; m < M; ++m) {
      h_bias_bc[static_cast<size_t>(m) + static_cast<size_t>(n) * static_cast<size_t>(M)] = h_b[n];
    }
  }

  float *d_a = nullptr;
  float *d_b = nullptr;
  float *d_c = nullptr;
  float *d_bias_bc = nullptr;
  checkCuda(cudaMalloc(&d_a, elems_x * sizeof(float)), "cudaMalloc d_a");
  checkCuda(cudaMalloc(&d_b, static_cast<size_t>(K) * static_cast<size_t>(N) * sizeof(float)),
             "cudaMalloc d_b");
  checkCuda(cudaMalloc(&d_c, elems_y * sizeof(float)), "cudaMalloc d_c");
  checkCuda(cudaMalloc(&d_bias_bc, elems_y * sizeof(float)), "cudaMalloc d_bias_bc");

  cublasHandle_t hdl{};
  checkCublas(cublasCreate(&hdl), "cublasCreate");

  // Host row-major M×K (lda=K) -> device column-major M×K (ldb=M)
  checkCublas(cublasSetMatrix(M, K, sizeof(float), h_x.data(), K, d_a, M), "SetMatrix X");
  // Host column-major K×N (lda=K) -> device column-major K×N (ldb=K)
  checkCublas(cublasSetMatrix(K, N, sizeof(float), h_wt.data(), K, d_b, K), "SetMatrix W^T");
  checkCublas(cublasSetMatrix(M, N, sizeof(float), h_bias_bc.data(), M, d_bias_bc, M),
              "SetMatrix bias_bc");

  const float alpha = 1.0f;
  const float beta_gemm = 0.0f;

  auto do_linear = [&]() {
    checkCublas(cublasSgemm(hdl, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_a, M, d_b, K,
                            &beta_gemm, d_c, M),
                "cublasSgemm");
    checkCublas(cublasSaxpy(hdl, static_cast<int>(elems_y), &alpha, d_bias_bc, 1, d_c, 1),
                "cublasSaxpy bias");
  };

  for (int w = 0; w < warmup; ++w) {
    do_linear();
  }
  checkCuda(cudaDeviceSynchronize(), "sync warmup");

  cudaEvent_t e0{}, e1{};
  checkCuda(cudaEventCreate(&e0), "event0");
  checkCuda(cudaEventCreate(&e1), "event1");
  checkCuda(cudaEventRecord(e0), "record0");
  for (int i = 0; i < iters; ++i) {
    do_linear();
  }
  checkCuda(cudaEventRecord(e1), "record1");
  checkCuda(cudaEventSynchronize(e1), "sync1");
  float ms_total = 0.f;
  checkCuda(cudaEventElapsedTime(&ms_total, e0, e1), "elapsed");

  const float ms_per_iter = ms_total / static_cast<float>(iters);
  const double flops =
      2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K) +
      static_cast<double>(M) * static_cast<double>(N);
  const double gflops = flops / (static_cast<double>(ms_per_iter) / 1000.0) / 1e9;

  printf("OK ms_per_iter %.6f gflops %.4f method cublas_sgemm_colmajor\n", ms_per_iter, gflops);

  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  cublasDestroy(hdl);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(d_bias_bc);
  return 0;
}
