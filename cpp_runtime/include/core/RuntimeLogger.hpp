#pragma once

#include "core/InferenceJob.hpp"
#include "core/InferenceResult.hpp"

#include <filesystem>
#include <mutex>
#include <string>

namespace edgeai {

class RuntimeLogger {
public:
    explicit RuntimeLogger(std::filesystem::path log_path = "logs/inference_metrics.jsonl");

    void logInference(const InferenceJob& job, const InferenceResult& result);
    const std::filesystem::path& logPath() const noexcept;

private:
    std::filesystem::path log_path_;
    std::mutex mutex_;
};

} // namespace edgeai

