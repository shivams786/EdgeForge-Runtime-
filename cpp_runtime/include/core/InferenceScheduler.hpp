#pragma once

#include "core/InferenceJob.hpp"
#include "core/InferenceResult.hpp"
#include "core/MetricsCollector.hpp"
#include "core/RuntimeConfig.hpp"
#include "core/RuntimeLogger.hpp"

#include <deque>
#include <memory>
#include <optional>
#include <vector>

namespace edgeai {

class InferenceScheduler {
public:
    explicit InferenceScheduler(RuntimeConfig config, RuntimeLogger* logger = nullptr, MetricsCollector* metrics = nullptr);

    void addJob(InferenceJob job);
    std::optional<InferenceResult> runNext();
    std::vector<InferenceResult> runAll();
    std::size_t getPendingJobCount() const noexcept;
    std::vector<std::string> pendingJobIdsInExecutionOrder() const;

private:
    RuntimeConfig config_;
    RuntimeLogger* logger_;
    MetricsCollector* metrics_;
    std::deque<InferenceJob> jobs_;
    std::size_t next_sequence_number_{0};

    std::optional<InferenceJob> popNextJob();
    InferenceResult executeJob(const InferenceJob& job) const;
};

} // namespace edgeai

