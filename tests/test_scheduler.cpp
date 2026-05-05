#include "test_common.hpp"

#include "core/InferenceScheduler.hpp"

namespace {

edgeai::InferenceJob makeJob(const std::string& id, int priority) {
    edgeai::InferenceJob job;
    job.job_id = id;
    job.model_path = "models/sample_model.onnx";
    job.input_path = "sample_inputs/input_vector.txt";
    job.priority = priority;
    return job;
}

} // namespace

EDGEAI_TEST(scheduler_fifo_preserves_insertion_order) {
    edgeai::RuntimeConfig config;
    config.schedule_policy = edgeai::SchedulePolicy::Fifo;
    edgeai::InferenceScheduler scheduler(config);
    scheduler.addJob(makeJob("a", 0));
    scheduler.addJob(makeJob("b", 100));
    scheduler.addJob(makeJob("c", 50));

    const auto ids = scheduler.pendingJobIdsInExecutionOrder();
    edgeai::test::require(ids.size() == 3, "expected three jobs");
    edgeai::test::require(ids[0] == "a" && ids[1] == "b" && ids[2] == "c", "FIFO order was not preserved");
}

EDGEAI_TEST(scheduler_priority_runs_highest_priority_first) {
    edgeai::RuntimeConfig config;
    config.schedule_policy = edgeai::SchedulePolicy::Priority;
    edgeai::InferenceScheduler scheduler(config);
    scheduler.addJob(makeJob("low", 1));
    scheduler.addJob(makeJob("high", 50));
    scheduler.addJob(makeJob("mid", 10));

    const auto ids = scheduler.pendingJobIdsInExecutionOrder();
    edgeai::test::require(ids[0] == "high", "highest priority should run first");
    edgeai::test::require(ids[1] == "mid", "middle priority should run second");
    edgeai::test::require(ids[2] == "low", "lowest priority should run last");
}

