#include "core/RuntimeLogger.hpp"

#include "core/TensorUtils.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace edgeai {

RuntimeLogger::RuntimeLogger(std::filesystem::path log_path)
    : log_path_(std::move(log_path)) {
    if (!log_path_.parent_path().empty()) {
        std::filesystem::create_directories(log_path_.parent_path());
    }
}

void RuntimeLogger::logInference(const InferenceJob& job, const InferenceResult& result) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!log_path_.parent_path().empty()) {
        std::filesystem::create_directories(log_path_.parent_path());
    }

    std::ofstream out(log_path_, std::ios::app);
    if (!out) {
        throw std::runtime_error("Failed to open log file: " + log_path_.string());
    }

    out << "{\"timestamp\":\"" << TensorUtils::jsonEscape(result.timestamp) << "\""
        << ",\"job_id\":\"" << TensorUtils::jsonEscape(job.job_id) << "\""
        << ",\"model_path\":\"" << TensorUtils::jsonEscape(job.model_path) << "\""
        << ",\"input_path\":\"" << TensorUtils::jsonEscape(job.input_path) << "\""
        << ",\"runner_type\":\"" << TensorUtils::jsonEscape(job.runner_type) << "\""
        << ",\"status\":\"" << (result.success ? "success" : "failure") << "\""
        << ",\"latency_ms\":" << std::fixed << std::setprecision(3) << result.inference_latency_ms
        << ",\"total_runtime_ms\":" << std::fixed << std::setprecision(3) << result.total_runtime_ms
        << ",\"output_size\":" << result.output.size()
        << ",\"error_message\":\"" << TensorUtils::jsonEscape(result.error_message) << "\""
        << "}\n";
}

const std::filesystem::path& RuntimeLogger::logPath() const noexcept {
    return log_path_;
}

} // namespace edgeai

