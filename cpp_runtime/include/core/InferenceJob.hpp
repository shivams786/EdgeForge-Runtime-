#pragma once

#include <chrono>
#include <string>

namespace edgeai {

struct InferenceJob {
    std::string job_id;
    std::string model_path;
    std::string input_path;
    std::string runner_type{"onnx"};
    int priority{0};
    std::chrono::system_clock::time_point created_at{std::chrono::system_clock::now()};
    std::size_t sequence_number{0};
};

} // namespace edgeai

