cmake_minimum_required(VERSION 3.16)
project(rmcompress VERSION 0.1.0 LANGUAGES CXX)

if(RMCOMPRESS_ENABLE_TENSORRT)
    enable_language(CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenVINO
find_package(OpenVINO REQUIRED)

option(RMCOMPRESS_ENABLE_TENSORRT "Enable optional TensorRT g_a sender backend" OFF)

if(RMCOMPRESS_ENABLE_TENSORRT)
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h)
    find_path(CUDA_RUNTIME_INCLUDE_DIR cuda_runtime_api.h
        PATHS
            /usr/local/cuda/include
            /usr/local/cuda-12.6/include
            /usr/local/cuda-12.6/targets/x86_64-linux/include
            /usr/local/cuda-12.5/include
            /usr/local/cuda-12.5/targets/x86_64-linux/include
            /usr/local/cuda-12.4/include
            /usr/local/cuda-12.4/targets/x86_64-linux/include
            /usr/local/cuda-12.3/include
            /usr/local/cuda-12.3/targets/x86_64-linux/include)
    find_library(TENSORRT_LIBRARY nvinfer)
    find_library(CUDART_LIBRARY cudart
        PATHS
            /usr/local/cuda/lib64
            /usr/local/cuda-12.6/lib64
            /usr/local/cuda-12.6/targets/x86_64-linux/lib
            /usr/local/cuda-12.5/lib64
            /usr/local/cuda-12.5/targets/x86_64-linux/lib
            /usr/local/cuda-12.4/lib64
            /usr/local/cuda-12.4/targets/x86_64-linux/lib
            /usr/local/cuda-12.3/lib64
            /usr/local/cuda-12.3/targets/x86_64-linux/lib
        NO_DEFAULT_PATH)
    if(NOT CUDART_LIBRARY)
        find_library(CUDART_LIBRARY cudart)
    endif()
    if(NOT TENSORRT_INCLUDE_DIR OR NOT CUDA_RUNTIME_INCLUDE_DIR OR
       NOT TENSORRT_LIBRARY OR NOT CUDART_LIBRARY)
        message(WARNING
            "RMCOMPRESS_ENABLE_TENSORRT=ON requested, but NvInfer.h/libnvinfer/libcudart "
            "were not all found. Building OpenVINO-only with TensorRT runtime error stubs.")
        set(RMCOMPRESS_TENSORRT_AVAILABLE OFF)
    else()
        set(RMCOMPRESS_TENSORRT_AVAILABLE ON)
    endif()
endif()

# --- shared library: librmcompress.so ---
add_library(rmcompress SHARED
    librmcompress/src/compressor.cpp
    librmcompress/src/preprocess.cpp
    librmcompress/src/mbt_encoder.cpp
    librmcompress/src/ga_encoder.cpp
    librmcompress/src/tensorrt_ga_encoder.cpp
    librmcompress/src/entropy_coder.cpp
    librmcompress/src/inference_engine.cpp
    librmcompress/src/protocol.cpp
    librmcompress/src/shm_ring.cpp
)
target_include_directories(rmcompress PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/librmcompress/include
)
target_link_libraries(rmcompress PUBLIC openvino::runtime)
if(RMCOMPRESS_TENSORRT_AVAILABLE)
    target_sources(rmcompress PRIVATE librmcompress/src/preprocess_cuda.cu)
    target_compile_definitions(rmcompress PRIVATE RMCOMPRESS_ENABLE_TENSORRT=1)
    target_include_directories(rmcompress PRIVATE
        ${TENSORRT_INCLUDE_DIR}
        ${CUDA_RUNTIME_INCLUDE_DIR}
    )
    target_link_libraries(rmcompress PRIVATE ${TENSORRT_LIBRARY} ${CUDART_LIBRARY})
    set_target_properties(rmcompress PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
# Resolve project root (where models/ lives) so the binary can be run from any CWD
get_filename_component(RM_PROJECT_ROOT "${CMAKE_SOURCE_DIR}/.." ABSOLUTE)
target_compile_definitions(rmcompress PRIVATE "RM_PROJECT_ROOT=${RM_PROJECT_ROOT}")

# --- CLI: rm_compress_cli ---
add_executable(rm_compress_cli
    cli/main.cpp
    serial/serial_tx.cpp
)
target_include_directories(rm_compress_cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(rm_compress_cli PRIVATE rmcompress)

# --- QVRF dry-run parity harness ---
add_executable(qvrf_dryrun_dump
    test/qvrf_dryrun_dump.cpp
)
target_include_directories(qvrf_dryrun_dump PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(qvrf_dryrun_dump PRIVATE rmcompress)

# --- CLI: rm_camera_capture ---
set(MVS_ROOT "/home/wyh/MVS_Linux_STD_V5.0.0_260421/MVS-5.0.0_x86_64_20260421/MVS"
    CACHE PATH "Hikrobot MVS SDK root")
set(MVS_INCLUDE_DIR "${MVS_ROOT}/include")
set(MVS_LIB_DIR "${MVS_ROOT}/lib/64")
if(EXISTS "${MVS_INCLUDE_DIR}/MvCameraControl.h" AND EXISTS "${MVS_LIB_DIR}/libMvCameraControl.so")
    add_executable(rm_camera_capture
        camera/camera_capture.cpp
    )
    target_include_directories(rm_camera_capture PRIVATE
        ${MVS_INCLUDE_DIR}
    )
    target_link_directories(rm_camera_capture PRIVATE ${MVS_LIB_DIR})
    target_link_libraries(rm_camera_capture PRIVATE rmcompress MvCameraControl pthread)
    set_target_properties(rm_camera_capture PROPERTIES
        BUILD_RPATH "${MVS_LIB_DIR}"
        INSTALL_RPATH "${MVS_LIB_DIR}"
    )
else()
    message(WARNING "MVS SDK not found at ${MVS_ROOT}; rm_camera_capture target disabled")
endif()

# --- tests (conditional: skip missing files) ---
enable_testing()

set(TEST_SOURCES "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_compressor.cpp")
    list(APPEND TEST_SOURCES test/test_compressor.cpp)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_entropy.cpp")
    list(APPEND TEST_SOURCES test/test_entropy.cpp)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_protocol.cpp")
    list(APPEND TEST_SOURCES test/test_protocol.cpp)
endif()

if(TEST_SOURCES)
    add_executable(test_rmcompress ${TEST_SOURCES})
    target_include_directories(test_rmcompress PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(test_rmcompress PRIVATE rmcompress)
    add_test(NAME rmcompress_tests COMMAND test_rmcompress)
    message(STATUS "Tests enabled: ${TEST_SOURCES}")
else()
    message(STATUS "No test sources found — skipping test target")
endif()
