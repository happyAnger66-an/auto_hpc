/*
 * C = A + B (FP32, row-major M x N contiguous), 与 CuTeDSL hello 语义一致。
 *
 * 实现：cublasScopy(A -> C) + cublasSaxpy(1, B, C) 即 C = A + B。
 * 说明：cuBLAS 经典 API 没有单调用“两向量相加写第三向量”，此为常见组合。
 *
 * 用法: ./elementwise_add_bench <M> <N> <warmup> <iters>
 * 输出: OK ms_per_iter <float> gflops <float> method cublas_scopy_saxpy（GFLOPS=M*N/time_s/1e9）
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
  if (argc != 5) {
    fprintf(stderr, "Usage: %s M N warmup iters\n", argv[0]);
    return 1;
  }
  const int M = std::atoi(argv[1]);
  const int N = std::atoi(argv[2]);
  const int warmup = std::atoi(argv[3]);
  const int iters = std::atoi(argv[4]);
  if (M <= 0 || N <= 0 || warmup < 0 || iters < 1) {
    fprintf(stderr, "Invalid M,N,warmup,iters\n");
    return 1;
  }

  const size_t elems = static_cast<size_t>(M) * static_cast<size_t>(N);
  const size_t bytes = elems * sizeof(float);

  float *d_a = nullptr;
  float *d_b = nullptr;
  float *d_c = nullptr;
  checkCuda(cudaMalloc(&d_a, bytes), "cudaMalloc d_a");
  checkCuda(cudaMalloc(&d_b, bytes), "cudaMalloc d_b");
  checkCuda(cudaMalloc(&d_c, bytes), "cudaMalloc d_c");

  std::vector<float> h(elems);
  for (size_t i = 0; i < elems; ++i) {
    h[i] = static_cast<float>(static_cast<int>(i % 997) * 1e-3f);
  }
  checkCuda(cudaMemcpy(d_a, h.data(), bytes, cudaMemcpyHostToDevice), "H2D A");
  for (size_t i = 0; i < elems; ++i) {
    h[i] = static_cast<float>(static_cast<int>(i % 503) * 1e-3f);
  }
  checkCuda(cudaMemcpy(d_b, h.data(), bytes, cudaMemcpyHostToDevice), "H2D B");

  cublasHandle_t handle{};
  checkCublas(cublasCreate(&handle), "cublasCreate");
  const float alpha = 1.0f;

  auto do_add = [&]() {
    checkCublas(cublasScopy(handle, static_cast<int>(elems), d_a, 1, d_c, 1),
                  "cublasScopy");
    checkCublas(cublasSaxpy(handle, static_cast<int>(elems), &alpha, d_b, 1, d_c, 1),
                  "cublasSaxpy");
  };

  for (int w = 0; w < warmup; ++w) {
    do_add();
  }
  checkCuda(cudaDeviceSynchronize(), "sync after warmup");

  cudaEvent_t e0{}, e1{};
  checkCuda(cudaEventCreate(&e0), "event0");
  checkCuda(cudaEventCreate(&e1), "event1");
  checkCuda(cudaEventRecord(e0), "record0");
  for (int i = 0; i < iters; ++i) {
    do_add();
  }
  checkCuda(cudaEventRecord(e1), "record1");
  checkCuda(cudaEventSynchronize(e1), "sync1");
  float ms_total = 0.f;
  checkCuda(cudaEventElapsedTime(&ms_total, e0, e1), "elapsed");

  const float ms_per_iter = ms_total / static_cast<float>(iters);
  // GFLOPS：逐元素加法 C=A+B，每个输出元素计 1 FLOP（一次 float 加）
  const double gflops =
      static_cast<double>(elems) / (static_cast<double>(ms_per_iter) / 1000.0) / 1e9;

  printf("OK ms_per_iter %.6f gflops %.4f method cublas_scopy_saxpy\n", ms_per_iter, gflops);

  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  cublasDestroy(handle);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return 0;
}
