set(EDGE_AI_HEADERS
    include/core/IModelRunner.hpp
    include/core/InferenceJob.hpp
    include/core/InferenceResult.hpp
    include/core/InferenceScheduler.hpp
    include/core/ModelRunnerFactory.hpp
    include/core/RuntimeConfig.hpp
    include/core/RuntimeLogger.hpp
    include/core/MetricsCollector.hpp
    include/core/TensorUtils.hpp
    include/runners/OnnxModelRunner.hpp
    include/cli/CliParser.hpp)

set(EDGE_AI_SOURCES
    src/core/InferenceScheduler.cpp
    src/core/RuntimeLogger.cpp
    src/core/MetricsCollector.cpp
    src/core/TensorUtils.cpp
    src/runners/OnnxModelRunner.cpp
    src/cli/CliParser.cpp)

add_library(edge_ai_runtime_lib ${EDGE_AI_HEADERS} ${EDGE_AI_SOURCES})
target_include_directories(edge_ai_runtime_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(EDGE_AI_ENABLE_ONNXRUNTIME)
    target_compile_definitions(edge_ai_runtime_lib PUBLIC EDGE_AI_ENABLE_ONNXRUNTIME=1)
    if(NOT ONNXRUNTIME_ROOT)
        message(FATAL_ERROR "ONNXRUNTIME_ROOT is not set. Configure with -DONNXRUNTIME_ROOT=/path/to/onnxruntime or set EDGE_AI_ENABLE_ONNXRUNTIME=OFF for parser/scheduler tests only.")
    else()
        target_include_directories(edge_ai_runtime_lib PUBLIC ${ONNXRUNTIME_ROOT}/include)
        find_library(ONNXRUNTIME_LIB
            NAMES onnxruntime onnxruntime.lib
            PATHS ${ONNXRUNTIME_ROOT}/lib ${ONNXRUNTIME_ROOT}/lib64
            NO_DEFAULT_PATH)
        if(NOT ONNXRUNTIME_LIB)
            message(FATAL_ERROR "Could not find ONNX Runtime library under ${ONNXRUNTIME_ROOT}/lib or lib64")
        endif()
        target_link_libraries(edge_ai_runtime_lib PUBLIC ${ONNXRUNTIME_LIB})
    endif()
else()
    target_compile_definitions(edge_ai_runtime_lib PUBLIC EDGE_AI_ENABLE_ONNXRUNTIME=0)
endif()

if(WIN32)
    target_link_libraries(edge_ai_runtime_lib PUBLIC psapi)
endif()

add_executable(edge_ai_runtime src/main.cpp)
target_link_libraries(edge_ai_runtime PRIVATE edge_ai_runtime_lib)
set_target_properties(edge_ai_runtime PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

if(WIN32 AND ONNXRUNTIME_ROOT AND ONNXRUNTIME_LIB)
    add_custom_command(TARGET edge_ai_runtime POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:edge_ai_runtime>
        COMMAND_EXPAND_LISTS)
endif()
