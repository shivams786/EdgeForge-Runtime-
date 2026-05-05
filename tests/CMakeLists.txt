add_executable(edge_ai_runtime_tests
    test_main.cpp
    test_scheduler.cpp
    test_logger.cpp
    test_metrics.cpp
    test_tensor_utils.cpp)

target_link_libraries(edge_ai_runtime_tests PRIVATE edge_ai_runtime_lib)
add_test(NAME edge_ai_runtime_tests COMMAND edge_ai_runtime_tests)

