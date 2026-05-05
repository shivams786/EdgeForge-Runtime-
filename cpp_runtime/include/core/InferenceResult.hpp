#pragma once

#include <string>
#include <vector>

namespace edgeai {

struct InferenceResult {
    std::string job_id;
    bool success{false};
    std::vector<float> output;
    std::string error_message;
    double model_load_time_ms{0.0};
    double inference_latency_ms{0.0};
    double total_runtime_ms{0.0};
    std::string timestamp;
};

} // namespace edgeai

