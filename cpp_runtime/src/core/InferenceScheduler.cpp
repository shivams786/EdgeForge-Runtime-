#include "core/InferenceScheduler.hpp"

#include "core/ModelRunnerFactory.hpp"
#include "core/TensorUtils.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace edgeai {

InferenceScheduler::InferenceScheduler(RuntimeConfig config, RuntimeLogger* logger, MetricsCollector* metrics)
    : config_(std::move(config)), logger_(logger), metrics_(metrics) {}

void InferenceScheduler::addJob(InferenceJob job) {
    job.sequence_number = next_sequence_number_++;
    if (job.job_id.empty()) {
        job.job_id = "job_" + std::to_string(job.sequence_number);
    }
    jobs_.push_back(std::move(job));
}

std::optional<InferenceJob> InferenceScheduler::popNextJob() {
    if (jobs_.empty()) {
        return std::nullopt;
    }

    if (config_.schedule_policy == SchedulePolicy::Fifo) {
        InferenceJob job = jobs_.front();
        jobs_.pop_front();
        return job;
    }

    auto best = std::max_element(jobs_.begin(), jobs_.end(), [](const InferenceJob& left, const InferenceJob& right) {
        if (left.priority == right.priority) {
            return left.sequence_number > right.sequence_number;
        }
        return left.priority < right.priority;
    });

    InferenceJob job = *best;
    jobs_.erase(best);
    return job;
}

InferenceResult InferenceScheduler::executeJob(const InferenceJob& job) const {
    const auto total_start = std::chrono::steady_clock::now();
    InferenceResult result;
    result.job_id = job.job_id;
    result.timestamp = TensorUtils::currentTimestampUtc();

    try {
        const auto input_data = TensorUtils::readFloatVector(job.input_path);
        auto runner = createModelRunner(job.runner_type);

        const auto load_start = std::chrono::steady_clock::now();
        runner->loadModel(job.model_path);
        const auto load_end = std::chrono::steady_clock::now();
        result.model_load_time_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();

        result = runner->runInference(job.job_id, input_data);
        result.model_load_time_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();
        result.success = true;
    } catch (const std::exception& ex) {
        result.success = false;
        result.error_message = ex.what();
    }

    const auto total_end = std::chrono::steady_clock::now();
    result.total_runtime_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    if (result.timestamp.empty()) {
        result.timestamp = TensorUtils::currentTimestampUtc();
    }
    return result;
}

std::optional<InferenceResult> InferenceScheduler::runNext() {
    auto job = popNextJob();
    if (!job) {
        return std::nullopt;
    }

    auto result = executeJob(*job);
    if (logger_) {
        logger_->logInference(*job, result);
    }
    if (metrics_) {
        metrics_->record(result);
    }
    return result;
}

std::vector<InferenceResult> InferenceScheduler::runAll() {
    std::vector<InferenceResult> results;
    while (auto result = runNext()) {
        results.push_back(*result);
    }
    return results;
}

std::size_t InferenceScheduler::getPendingJobCount() const noexcept {
    return jobs_.size();
}

std::vector<std::string> InferenceScheduler::pendingJobIdsInExecutionOrder() const {
    std::vector<InferenceJob> copy(jobs_.begin(), jobs_.end());
    if (config_.schedule_policy == SchedulePolicy::Priority) {
        std::stable_sort(copy.begin(), copy.end(), [](const InferenceJob& left, const InferenceJob& right) {
            if (left.priority == right.priority) {
                return left.sequence_number < right.sequence_number;
            }
            return left.priority > right.priority;
        });
    }

    std::vector<std::string> ids;
    ids.reserve(copy.size());
    for (const auto& job : copy) {
        ids.push_back(job.job_id);
    }
    return ids;
}

} // namespace edgeai
