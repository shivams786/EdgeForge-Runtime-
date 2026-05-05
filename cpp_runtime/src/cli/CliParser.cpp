#include "cli/CliParser.hpp"

#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace edgeai {

namespace {

std::string readFile(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Failed to open jobs file: " + path.string());
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

std::optional<std::string> extractStringField(const std::string& object, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch match;
    if (std::regex_search(object, match, pattern)) {
        return match[1].str();
    }
    return std::nullopt;
}

int extractIntField(const std::string& object, const std::string& key, int default_value) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*(-?\\d+)");
    std::smatch match;
    if (std::regex_search(object, match, pattern)) {
        return std::stoi(match[1].str());
    }
    return default_value;
}

std::string requireStringField(const std::string& object, const std::string& key) {
    auto value = extractStringField(object, key);
    if (!value || value->empty()) {
        throw std::runtime_error("Job entry is missing required field: " + key);
    }
    return *value;
}

} // namespace

CliOptions CliParser::parse(int argc, char** argv) {
    CliOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto requireValue = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing value for " + flag);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            options.help = true;
        } else if (arg == "--model") {
            options.model_path = requireValue(arg);
        } else if (arg == "--input") {
            options.input_path = requireValue(arg);
        } else if (arg == "--runner") {
            options.runner_type = requireValue(arg);
        } else if (arg == "--jobs") {
            options.jobs_path = requireValue(arg);
        } else if (arg == "--schedule") {
            options.schedule_policy = parseSchedulePolicy(requireValue(arg));
        } else if (arg == "--metrics") {
            options.metrics = true;
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }

    if (!options.help) {
        const bool single_mode = options.model_path.has_value() || options.input_path.has_value();
        const bool batch_mode = options.jobs_path.has_value();
        if (single_mode && batch_mode) {
            throw std::invalid_argument("Use either --model/--input or --jobs, not both");
        }
        if (single_mode && (!options.model_path || !options.input_path)) {
            throw std::invalid_argument("Single inference requires both --model and --input");
        }
        if (!single_mode && !batch_mode) {
            throw std::invalid_argument("No work specified. Use --model/--input, --jobs, or --help");
        }
    }

    return options;
}

std::string CliParser::helpText(const std::string& executable_name) {
    std::ostringstream out;
    out << "Edge AI Inference Orchestration Runtime\n\n"
        << "Usage:\n"
        << "  " << executable_name << " --model models/sample_model.onnx --input sample_inputs/input_vector.txt --runner onnx\n"
        << "  " << executable_name << " --jobs configs/sample_jobs.json --schedule fifo --metrics\n"
        << "  " << executable_name << " --jobs configs/sample_jobs.json --schedule priority --metrics\n\n"
        << "Options:\n"
        << "  --model <path>      ONNX model for single inference\n"
        << "  --input <path>      Numeric input vector file for single inference\n"
        << "  --runner <type>     Runner backend, currently: onnx\n"
        << "  --jobs <path>       JSON array of inference jobs\n"
        << "  --schedule <policy> fifo or priority\n"
        << "  --metrics           Print metrics summary\n"
        << "  --help              Show this help\n";
    return out.str();
}

std::vector<InferenceJob> loadJobsFromJsonFile(const std::filesystem::path& path) {
    const std::string content = readFile(path);
    const std::regex object_pattern("\\{[^\\{\\}]*\\}");
    std::sregex_iterator it(content.begin(), content.end(), object_pattern);
    std::sregex_iterator end;

    std::vector<InferenceJob> jobs;
    for (; it != end; ++it) {
        const std::string object = it->str();
        InferenceJob job;
        job.job_id = requireStringField(object, "job_id");
        job.model_path = requireStringField(object, "model_path");
        job.input_path = requireStringField(object, "input_path");
        job.runner_type = extractStringField(object, "runner_type").value_or("onnx");
        job.priority = extractIntField(object, "priority", 0);
        jobs.push_back(std::move(job));
    }

    if (jobs.empty()) {
        throw std::runtime_error("Jobs file did not contain any job objects: " + path.string());
    }

    return jobs;
}

} // namespace edgeai

