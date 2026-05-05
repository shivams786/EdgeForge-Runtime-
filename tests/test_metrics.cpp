#include "test_common.hpp"

#include "core/MetricsCollector.hpp"

EDGEAI_TEST(metrics_collector_calculates_latency_and_failures) {
    edgeai::MetricsCollector collector;

    edgeai::InferenceResult ok1;
    ok1.success = true;
    ok1.inference_latency_ms = 10.0;
    ok1.total_runtime_ms = 20.0;
    collector.record(ok1);

    edgeai::InferenceResult ok2;
    ok2.success = true;
    ok2.inference_latency_ms = 30.0;
    ok2.total_runtime_ms = 40.0;
    collector.record(ok2);

    edgeai::InferenceResult fail;
    fail.success = false;
    fail.total_runtime_ms = 10.0;
    collector.record(fail);

    const auto snapshot = collector.snapshot();
    edgeai::test::require(snapshot.total_jobs == 3, "total_jobs mismatch");
    edgeai::test::require(snapshot.successful_jobs == 2, "successful_jobs mismatch");
    edgeai::test::require(snapshot.failed_jobs == 1, "failed_jobs mismatch");
    edgeai::test::require(snapshot.average_latency_ms == 20.0, "average latency mismatch");
    edgeai::test::require(snapshot.min_latency_ms == 10.0, "min latency mismatch");
    edgeai::test::require(snapshot.max_latency_ms == 30.0, "max latency mismatch");
    edgeai::test::require(snapshot.failure_rate > 0.33 && snapshot.failure_rate < 0.34, "failure rate mismatch");
}

