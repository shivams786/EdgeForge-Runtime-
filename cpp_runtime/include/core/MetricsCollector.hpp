#pragma once

#include "core/InferenceResult.hpp"

#include <cstddef>
#include <string>

namespace edgeai {

struct MetricsSnapshot {
    std::size_t total_jobs{0};
    std::size_t successful_jobs{0};
    std::size_t failed_jobs{0};
    double average_latency_ms{0.0};
    double min_latency_ms{0.0};
    double max_latency_ms{0.0};
    double approximate_throughput_jobs_per_sec{0.0};
    double failure_rate{0.0};
    double memory_usage_mb{0.0};
};

class MetricsCollector {
public:
    void record(const InferenceResult& result);
    MetricsSnapshot snapshot() const;
    std::string summaryString() const;

private:
    std::size_t total_jobs_{0};
    std::size_t successful_jobs_{0};
    std::size_t failed_jobs_{0};
    double total_latency_ms_{0.0};
    double min_latency_ms_{0.0};
    double max_latency_ms_{0.0};
    double total_runtime_ms_{0.0};
};

} // namespace edgeai

