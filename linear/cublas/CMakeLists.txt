# FP32 Linear Y = X @ W^T + b via cuBLAS Sgemm（设备端列主序）+ Saxpy（bias 广播）。
cmake_minimum_required(VERSION 3.18)
project(cublas_linear_bench LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(linear_bench linear_bench.cu)
target_link_libraries(linear_bench PRIVATE CUDA::cudart CUDA::cublas)
set_target_properties(linear_bench PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CXX_STANDARD 17
)
