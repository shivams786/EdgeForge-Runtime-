#include "cli/CliParser.hpp"
#include "core/InferenceScheduler.hpp"
#include "core/MetricsCollector.hpp"
#include "core/RuntimeLogger.hpp"

#include <exception>
#include <iostream>

namespace {

void printResult(const edgeai::InferenceResult& result) {
    std::cout << result.job_id << ": " << (result.success ? "success" : "failure")
              << ", latency_ms=" << result.inference_latency_ms
              << ", total_runtime_ms=" << result.total_runtime_ms
              << ", output_size=" << result.output.size();
    if (!result.error_message.empty()) {
        std::cout << ", error=\"" << result.error_message << "\"";
    }
    std::cout << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = edgeai::CliParser::parse(argc, argv);
        if (options.help) {
            std::cout << edgeai::CliParser::helpText(argc > 0 ? argv[0] : "edge_ai_runtime");
            return 0;
        }

        edgeai::RuntimeConfig config;
        config.schedule_policy = options.schedule_policy;
        config.print_metrics = options.metrics;

        edgeai::RuntimeLogger logger(config.log_path);
        edgeai::MetricsCollector metrics;
        edgeai::InferenceScheduler scheduler(config, &logger, &metrics);

        if (options.jobs_path) {
            for (auto& job : edgeai::loadJobsFromJsonFile(*options.jobs_path)) {
                scheduler.addJob(std::move(job));
            }
        } else {
            edgeai::InferenceJob job;
            job.job_id = "job_single";
            job.model_path = *options.model_path;
            job.input_path = *options.input_path;
            job.runner_type = options.runner_type;
            scheduler.addJob(std::move(job));
        }

        const auto results = scheduler.runAll();
        for (const auto& result : results) {
            printResult(result);
        }

        if (options.metrics) {
            std::cout << metrics.summaryString();
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "edge_ai_runtime: " << ex.what() << '\n';
        std::cerr << edgeai::CliParser::helpText(argc > 0 ? argv[0] : "edge_ai_runtime");
        return 1;
    }
}

