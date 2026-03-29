# FP32 element-wise add benchmark via cuDNN OpTensor ADD.
cmake_minimum_required(VERSION 3.18)
project(cudnn_elementwise_add_bench LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# 可通过 -DCUDNN_ROOT=/path/to/cudnn 指定；亦支持环境变量 CUDNN_PATH / CUDNN_ROOT
if(NOT DEFINED CUDNN_ROOT AND DEFINED ENV{CUDNN_PATH})
  set(CUDNN_ROOT "$ENV{CUDNN_PATH}")
endif()
if(NOT DEFINED CUDNN_ROOT AND DEFINED ENV{CUDNN_ROOT})
  set(CUDNN_ROOT "$ENV{CUDNN_ROOT}")
endif()

# 头文件：tarball 安装在 ${CUDNN_ROOT}/include；发行版常在 /usr/include
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

# 库：Debian/Ubuntu 常见为 /usr/lib/x86_64-linux-gnu/libcudnn.so.9（无 libcudnn.so 时仅 find_library(cudnn) 会失败）
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
    "未找到 cuDNN（需要 cudnn.h 与 libcudnn）。\n"
    "  已搜索头文件: CUDNN_ROOT/include、/usr/include 等。\n"
    "  已搜索库: CUDNN_ROOT/lib*、/usr/lib/x86_64-linux-gnu（含 libcudnn.so.9）等。\n"
    "  若使用 NVIDIA 压缩包，请设置: -DCUDNN_ROOT=/path/to/cudnn")
endif()

message(STATUS "cuDNN include: ${CUDNN_INCLUDE_DIR}")
message(STATUS "cuDNN library: ${CUDNN_LIBRARY}")

add_executable(elementwise_add_bench elementwise_add_bench.cu)
target_include_directories(elementwise_add_bench PRIVATE "${CUDNN_INCLUDE_DIR}")
target_link_libraries(elementwise_add_bench PRIVATE CUDA::cudart "${CUDNN_LIBRARY}")
set_target_properties(elementwise_add_bench PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CXX_STANDARD 17
)
