# 独立微基准：验证 cp.async.ca.shared.global（4B）+ commit / wait_group（无 CuTeDSL）。
# RTX 4070 使用 sm_89；其它 GPU 可 cmake -DCMAKE_CUDA_ARCHITECTURES=80 等覆盖。
cmake_minimum_required(VERSION 3.18)
project(cpasync_microbench LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# 默认 Ada sm_89（4070）；未设置时部分 CMake 版本不会把 arch 传给 nvcc，导致 ptxas 报 cp.async 需 sm_80+。
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

add_executable(cpasync_verify cpasync_verify.cu)
target_link_libraries(cpasync_verify PRIVATE CUDA::cudart)
set_target_properties(cpasync_verify PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CXX_STANDARD 17
  CUDA_STANDARD 17
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)
