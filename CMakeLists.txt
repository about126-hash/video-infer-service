cmake_minimum_required(VERSION 3.16)
project(video_infer_service)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 依赖路径：换机器时只需改这里，或 cmake -DTRT_ROOT=... / export TENSORRT_ROOT=...
set(TRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH "TensorRT 根目录（含 include/ 与 lib/）")
if(NOT TRT_ROOT)
    set(TRT_ROOT "/usr/local/TensorRT")
endif()

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-9)
enable_language(CUDA)

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

include_directories(${TRT_ROOT}/include/)
link_directories(${TRT_ROOT}/lib/)

include_directories(${PROJECT_SOURCE_DIR}/src/)
include_directories(${PROJECT_SOURCE_DIR}/common/)

# 找OpenCV
find_package(OpenCV REQUIRED)

# 头文件搜索路径
include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

# 生成可执行文件
add_executable(server
    src/server.cpp
    common/detector.hpp
    common/postprocess.cpp
    common/preprocess.cu
    common/common.hpp
)

add_executable(client
    src/client.cpp
)

# 链接OpenCV，httplib不需要链接库
target_link_libraries(server
    ${OpenCV_LIBS}
    pthread
    nvinfer
    cudart
)

# client 不依赖 CUDA / TensorRT
target_link_libraries(client
    ${OpenCV_LIBS}
    pthread
)
