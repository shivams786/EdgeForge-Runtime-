#pragma once

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace edgeai {

enum class SchedulePolicy {
    Fifo,
    Priority
};

struct RuntimeConfig {
    SchedulePolicy schedule_policy{SchedulePolicy::Fifo};
    bool print_metrics{false};
    std::filesystem::path log_path{"logs/inference_metrics.jsonl"};
    std::vector<int64_t> input_shape{1, 4};
};

inline SchedulePolicy parseSchedulePolicy(const std::string& value) {
    if (value == "fifo") {
        return SchedulePolicy::Fifo;
    }
    if (value == "priority") {
        return SchedulePolicy::Priority;
    }
    throw std::invalid_argument("Unsupported schedule policy: " + value);
}

inline std::string schedulePolicyToString(SchedulePolicy policy) {
    return policy == SchedulePolicy::Priority ? "priority" : "fifo";
}

} // namespace edgeai
