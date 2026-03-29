# FP32 element-wise add benchmark via cuBLAS (Scopy + Saxpy).
cmake_minimum_required(VERSION 3.18)
project(cublas_elementwise_add_bench LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(elementwise_add_bench elementwise_add_bench.cu)
target_link_libraries(elementwise_add_bench PRIVATE CUDA::cudart CUDA::cublas)
set_target_properties(elementwise_add_bench PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CXX_STANDARD 17
)
