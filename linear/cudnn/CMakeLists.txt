# FP32 linear Y = X @ W^T + b via cuDNN 1x1 conv + AddTensor (bias).
cmake_minimum_required(VERSION 3.18)
project(cudnn_linear_bench LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CUDNN_ROOT AND DEFINED ENV{CUDNN_PATH})
  set(CUDNN_ROOT "$ENV{CUDNN_PATH}")
endif()
if(NOT DEFINED CUDNN_ROOT AND DEFINED ENV{CUDNN_ROOT})
  set(CUDNN_ROOT "$ENV{CUDNN_ROOT}")
endif()

find_path(
  CUDNN_INCLUDE_DIR
  NAMES cudnn.h
  HINTS "${CUDNN_ROOT}"
  PATHS
    /usr/include
    /usr/local/include
    "${CUDAToolkit_INCLUDE_DIRS}"
  PATH_SUFFIXES include
  DOC "cuDNN include directory")

find_library(
  CUDNN_LIBRARY
  NAMES cudnn cudnn9 libcudnn libcudnn.so.9
  HINTS "${CUDNN_ROOT}"
  PATHS
    /usr/lib/x86_64-linux-gnu
    /usr/lib64
    /usr/lib
    "${CUDAToolkit_LIBRARY_DIR}"
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
  DOC "cuDNN library")

if(NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY)
  message(FATAL_ERROR
    "未找到 cuDNN。可设置 -DCUDNN_ROOT= 或安装系统包 libcudnn9。")
endif()

message(STATUS "cuDNN include: ${CUDNN_INCLUDE_DIR}")
message(STATUS "cuDNN library: ${CUDNN_LIBRARY}")

add_executable(linear_bench linear_bench.cu)
target_include_directories(linear_bench PRIVATE "${CUDNN_INCLUDE_DIR}")
target_link_libraries(linear_bench PRIVATE CUDA::cudart "${CUDNN_LIBRARY}")
set_target_properties(linear_bench PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CXX_STANDARD 17
)
