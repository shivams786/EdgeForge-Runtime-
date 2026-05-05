#include "test_common.hpp"

#include "core/RuntimeLogger.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>

EDGEAI_TEST(logger_writes_jsonl_entry) {
    const auto path = std::filesystem::temp_directory_path() / "edge_ai_runtime_logger_test.jsonl";
    std::filesystem::remove(path);

    edgeai::RuntimeLogger logger(path);
    edgeai::InferenceJob job;
    job.job_id = "job_log";
    job.model_path = "model.onnx";
    job.input_path = "input.txt";
    job.runner_type = "onnx";

    edgeai::InferenceResult result;
    result.job_id = job.job_id;
    result.success = true;
    result.timestamp = "2026-05-04T10:15:30Z";
    result.inference_latency_ms = 4.82;
    result.total_runtime_ms = 18.33;
    result.output = {1.0F, 2.0F};

    logger.logInference(job, result);

    std::ifstream input(path);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    const auto line = buffer.str();
    edgeai::test::require(line.find("\"job_id\":\"job_log\"") != std::string::npos, "job_id missing from log");
    edgeai::test::require(line.find("\"status\":\"success\"") != std::string::npos, "status missing from log");
    edgeai::test::require(line.find("\"output_size\":2") != std::string::npos, "output size missing from log");

    std::filesystem::remove(path);
}

