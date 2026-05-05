#pragma once

#include "core/InferenceJob.hpp"
#include "core/RuntimeConfig.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace edgeai {

struct CliOptions {
    bool help{false};
    bool metrics{false};
    std::optional<std::string> model_path;
    std::optional<std::string> input_path;
    std::string runner_type{"onnx"};
    std::optional<std::string> jobs_path;
    SchedulePolicy schedule_policy{SchedulePolicy::Fifo};
};

class CliParser {
public:
    static CliOptions parse(int argc, char** argv);
    static std::string helpText(const std::string& executable_name);
};

std::vector<InferenceJob> loadJobsFromJsonFile(const std::filesystem::path& path);

} // namespace edgeai
