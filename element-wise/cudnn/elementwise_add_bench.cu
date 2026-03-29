/*
 * C = A + B (FP32), 张量形状 1 x 1 x M x N (NCHW)，线性存储与行主序 M x N 一致。
 *
 * 实现：cudnnOpTensor ADD，C = 1*A + 1*B + 0*C。
 *
 * 用法: ./elementwise_add_bench <M> <N> <warmup> <iters>
 * 输出: OK ms_per_iter <float> gflops <float> method cudnn_op_tensor_add
 */

#include <cuda_runtime.h>
#include <cudnn.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

static void checkCuda(cudaError_t e, const char *msg) {
  if (e != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
    std::exit(1);
  }
}

static void checkCudnn(cudnnStatus_t s, const char *msg) {
  if (s != CUDNN_STATUS_SUCCESS) {
    fprintf(stderr, "%s: cudnn error %d\n", msg, (int)s);
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

  cudnnHandle_t cudnn{};
  checkCudnn(cudnnCreate(&cudnn), "cudnnCreate");

  cudnnTensorDescriptor_t da{}, db{}, dc{};
  checkCudnn(cudnnCreateTensorDescriptor(&da), "create da");
  checkCudnn(cudnnCreateTensorDescriptor(&db), "create db");
  checkCudnn(cudnnCreateTensorDescriptor(&dc), "create dc");

  // 1,1,M,N 与行主序 MxN 连续内存一致（W 为最内维）
  checkCudnn(cudnnSetTensor4dDescriptor(da, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, M, N),
             "set da");
  checkCudnn(cudnnSetTensor4dDescriptor(db, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, M, N),
             "set db");
  checkCudnn(cudnnSetTensor4dDescriptor(dc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, M, N),
             "set dc");

  cudnnOpTensorDescriptor_t op{};
  checkCudnn(cudnnCreateOpTensorDescriptor(&op), "create op");
  checkCudnn(cudnnSetOpTensorDescriptor(op, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT,
                                        CUDNN_NOT_PROPAGATE_NAN),
             "set op");

  const float alpha1 = 1.0f;
  const float alpha2 = 1.0f;
  const float beta = 0.0f;

  auto do_add = [&]() {
    checkCudnn(cudnnOpTensor(cudnn, op, &alpha1, da, d_a, &alpha2, db, d_b, &beta, dc, d_c),
               "cudnnOpTensor");
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
  const double gflops =
      static_cast<double>(elems) / (static_cast<double>(ms_per_iter) / 1000.0) / 1e9;

  printf("OK ms_per_iter %.6f gflops %.4f method cudnn_op_tensor_add\n", ms_per_iter, gflops);

  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  cudnnDestroyOpTensorDescriptor(op);
  cudnnDestroyTensorDescriptor(da);
  cudnnDestroyTensorDescriptor(db);
  cudnnDestroyTensorDescriptor(dc);
  cudnnDestroy(cudnn);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return 0;
}
