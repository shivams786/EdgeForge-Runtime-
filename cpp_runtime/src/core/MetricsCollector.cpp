#include "core/MetricsCollector.hpp"

#include "core/TensorUtils.hpp"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>

namespace edgeai {

void MetricsCollector::record(const InferenceResult& result) {
    ++total_jobs_;
    total_runtime_ms_ += result.total_runtime_ms;

    if (result.success) {
        ++successful_jobs_;
        total_latency_ms_ += result.inference_latency_ms;
        if (successful_jobs_ == 1) {
            min_latency_ms_ = result.inference_latency_ms;
            max_latency_ms_ = result.inference_latency_ms;
        } else {
            min_latency_ms_ = std::min(min_latency_ms_, result.inference_latency_ms);
            max_latency_ms_ = std::max(max_latency_ms_, result.inference_latency_ms);
        }
    } else {
        ++failed_jobs_;
    }
}

MetricsSnapshot MetricsCollector::snapshot() const {
    MetricsSnapshot metrics;
    metrics.total_jobs = total_jobs_;
    metrics.successful_jobs = successful_jobs_;
    metrics.failed_jobs = failed_jobs_;
    metrics.average_latency_ms = successful_jobs_ == 0 ? 0.0 : total_latency_ms_ / static_cast<double>(successful_jobs_);
    metrics.min_latency_ms = successful_jobs_ == 0 ? 0.0 : min_latency_ms_;
    metrics.max_latency_ms = successful_jobs_ == 0 ? 0.0 : max_latency_ms_;
    metrics.approximate_throughput_jobs_per_sec =
        total_runtime_ms_ <= 0.0 ? 0.0 : static_cast<double>(total_jobs_) / (total_runtime_ms_ / 1000.0);
    metrics.failure_rate = total_jobs_ == 0 ? 0.0 : static_cast<double>(failed_jobs_) / static_cast<double>(total_jobs_);
    metrics.memory_usage_mb = TensorUtils::getCurrentMemoryUsageMb();
    return metrics;
}

std::string MetricsCollector::summaryString() const {
    const auto metrics = snapshot();
    std::ostringstream out;
    out << "Runtime Metrics\n"
        << "  total_jobs: " << metrics.total_jobs << '\n'
        << "  successful_jobs: " << metrics.successful_jobs << '\n'
        << "  failed_jobs: " << metrics.failed_jobs << '\n'
        << std::fixed << std::setprecision(3)
        << "  average_latency_ms: " << metrics.average_latency_ms << '\n'
        << "  min_latency_ms: " << metrics.min_latency_ms << '\n'
        << "  max_latency_ms: " << metrics.max_latency_ms << '\n'
        << "  approximate_throughput_jobs_per_sec: " << metrics.approximate_throughput_jobs_per_sec << '\n'
        << "  failure_rate: " << metrics.failure_rate << '\n'
        << "  memory_usage_mb: " << metrics.memory_usage_mb << '\n';
    return out.str();
}

} // namespace edgeai
