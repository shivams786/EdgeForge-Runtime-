cmake_minimum_required(VERSION 3.16)

project(EdgeAIInferenceRuntime
    VERSION 1.0.0
    DESCRIPTION "Edge AI Inference Orchestration Runtime"
    LANGUAGES CXX)

option(BUILD_TESTS "Build unit tests" ON)
option(EDGE_AI_ENABLE_ONNXRUNTIME "Enable ONNX Runtime backend" ON)
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime installation root")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_subdirectory(cpp_runtime)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

