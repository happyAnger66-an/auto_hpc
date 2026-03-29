/*
 * FP32 Linear: Y = X @ W^T + b
 *
 * X: [M, K], W: [N, K], b: [N], Y: [M, N]（行主序 contiguous，与 PyTorch nn.Linear 一致）
 *
 * 实现：将 matmul 视为 NCHW 下 1x1 卷积（M 为 N、K 为 C），再 cudnnAddTensor 加 bias。
 *
 * 用法: ./linear_bench M N K warmup iters
 * 输出: OK ms_per_iter <float> gflops <float> method cudnn_conv1x1_addtensor
 *
 * GFLOPS = (2*M*N*K + M*N) / time_s / 1e9
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

  float *d_x = nullptr;
  float *d_w = nullptr;
  float *d_b = nullptr;
  float *d_y = nullptr;
  checkCuda(cudaMalloc(&d_x, elems_x * sizeof(float)), "cudaMalloc d_x");
  checkCuda(cudaMalloc(&d_w, elems_w * sizeof(float)), "cudaMalloc d_w");
  checkCuda(cudaMalloc(&d_b, elems_b * sizeof(float)), "cudaMalloc d_b");
  checkCuda(cudaMalloc(&d_y, elems_y * sizeof(float)), "cudaMalloc d_y");

  std::vector<float> hx(elems_x);
  std::vector<float> hw(elems_w);
  std::vector<float> hb(elems_b);
  for (size_t i = 0; i < elems_x; ++i) {
    hx[i] = static_cast<float>(static_cast<int>(i % 911) * 1e-3f);
  }
  for (size_t i = 0; i < elems_w; ++i) {
    hw[i] = static_cast<float>(static_cast<int>(i % 733) * 1e-3f);
  }
  for (size_t i = 0; i < elems_b; ++i) {
    hb[i] = static_cast<float>(static_cast<int>(i % 401) * 1e-3f);
  }
  checkCuda(cudaMemcpy(d_x, hx.data(), elems_x * sizeof(float), cudaMemcpyHostToDevice), "H2D x");
  checkCuda(cudaMemcpy(d_w, hw.data(), elems_w * sizeof(float), cudaMemcpyHostToDevice), "H2D w");
  checkCuda(cudaMemcpy(d_b, hb.data(), elems_b * sizeof(float), cudaMemcpyHostToDevice), "H2D b");

  cudnnHandle_t cudnn{};
  checkCudnn(cudnnCreate(&cudnn), "cudnnCreate");

  cudnnTensorDescriptor_t xDesc{}, yDesc{}, bDesc{};
  cudnnFilterDescriptor_t wDesc{};
  cudnnConvolutionDescriptor_t convDesc{};

  checkCudnn(cudnnCreateTensorDescriptor(&xDesc), "xDesc");
  checkCudnn(cudnnCreateTensorDescriptor(&yDesc), "yDesc");
  checkCudnn(cudnnCreateTensorDescriptor(&bDesc), "bDesc");
  checkCudnn(cudnnCreateFilterDescriptor(&wDesc), "wDesc");
  checkCudnn(cudnnCreateConvolutionDescriptor(&convDesc), "convDesc");

  // X: [M, K, 1, 1] NCHW
  checkCudnn(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, M, K, 1, 1),
             "set xDesc");
  // W: [N, K, 1, 1]
  checkCudnn(cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, N, K, 1, 1),
             "set wDesc");
  // 1x1 conv, stride 1, pad 0
  checkCudnn(cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION,
                                             CUDNN_DATA_FLOAT),
             "set conv");

  int out_n = 0, out_c = 0, out_h = 0, out_w = 0;
  checkCudnn(cudnnGetConvolution2dForwardOutputDim(convDesc, xDesc, wDesc, &out_n, &out_c, &out_h, &out_w),
             "get out dim");
  if (out_n != M || out_c != N || out_h != 1 || out_w != 1) {
    fprintf(stderr, "Unexpected conv output shape %d %d %d %d\n", out_n, out_c, out_h, out_w);
    return 1;
  }

  checkCudnn(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, M, N, 1, 1),
             "set yDesc");
  checkCudnn(cudnnSetTensor4dDescriptor(bDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, N, 1, 1),
             "set bDesc");

  const cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  size_t ws_size = 0;
  checkCudnn(cudnnGetConvolutionForwardWorkspaceSize(cudnn, xDesc, wDesc, convDesc, yDesc, algo, &ws_size),
             "workspace size");

  void *d_ws = nullptr;
  if (ws_size > 0) {
    checkCuda(cudaMalloc(&d_ws, ws_size), "cudaMalloc workspace");
  }

  const float alpha = 1.0f;
  const float beta0 = 0.0f;
  const float beta1 = 1.0f;

  auto do_linear = [&]() {
    checkCudnn(cudnnConvolutionForward(cudnn, &alpha, xDesc, d_x, wDesc, d_w, convDesc, algo, d_ws,
                                       ws_size, &beta0, yDesc, d_y),
               "cudnnConvolutionForward");
    checkCudnn(cudnnAddTensor(cudnn, &alpha, bDesc, d_b, &beta1, yDesc, d_y), "cudnnAddTensor");
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

  printf("OK ms_per_iter %.6f gflops %.4f method cudnn_conv1x1_addtensor\n", ms_per_iter, gflops);

  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  if (d_ws) {
    cudaFree(d_ws);
  }
  cudnnDestroyConvolutionDescriptor(convDesc);
  cudnnDestroyFilterDescriptor(wDesc);
  cudnnDestroyTensorDescriptor(bDesc);
  cudnnDestroyTensorDescriptor(yDesc);
  cudnnDestroyTensorDescriptor(xDesc);
  cudnnDestroy(cudnn);
  cudaFree(d_x);
  cudaFree(d_w);
  cudaFree(d_b);
  cudaFree(d_y);
  return 0;
}
