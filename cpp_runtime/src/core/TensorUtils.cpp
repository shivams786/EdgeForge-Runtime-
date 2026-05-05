#include "core/TensorUtils.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <ctime>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#else
#include <unistd.h>
#endif

namespace edgeai::TensorUtils {

std::vector<float> parseFloatVectorText(const std::string& content) {
    std::string normalized = content;
    for (char& ch : normalized) {
        if (ch == '[' || ch == ']' || ch == ',' || ch == '\n' || ch == '\r' || ch == '\t') {
            ch = ' ';
        }
    }

    std::istringstream stream(normalized);
    std::vector<float> values;
    float value = 0.0F;
    while (stream >> value) {
        values.push_back(value);
    }

    if (values.empty()) {
        throw std::runtime_error("Input did not contain any parseable float values");
    }
    return values;
}

std::vector<float> readFloatVector(const std::filesystem::path& input_path) {
    if (!std::filesystem::exists(input_path)) {
        throw std::runtime_error("Input file does not exist: " + input_path.string());
    }

    std::ifstream input(input_path);
    if (!input) {
        throw std::runtime_error("Failed to open input file: " + input_path.string());
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    return parseFloatVectorText(buffer.str());
}

std::string currentTimestampUtc() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);

    std::tm utc_tm{};
#ifdef _WIN32
    gmtime_s(&utc_tm, &now_time);
#else
    gmtime_r(&now_time, &utc_tm);
#endif

    std::ostringstream out;
    out << std::put_time(&utc_tm, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

std::string jsonEscape(const std::string& value) {
    std::ostringstream out;
    for (const char ch : value) {
        switch (ch) {
        case '\\':
            out << "\\\\";
            break;
        case '"':
            out << "\\\"";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            if (static_cast<unsigned char>(ch) < 0x20) {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<int>(static_cast<unsigned char>(ch));
            } else {
                out << ch;
            }
        }
    }
    return out.str();
}

double getCurrentMemoryUsageMb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX counters{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters), sizeof(counters))) {
        return static_cast<double>(counters.WorkingSetSize) / (1024.0 * 1024.0);
    }
    return 0.0;
#else
    std::ifstream status("/proc/self/status");
    std::string key;
    while (status >> key) {
        if (key == "VmRSS:") {
            double kb = 0.0;
            std::string unit;
            status >> kb >> unit;
            return kb / 1024.0;
        }
        std::string rest_of_line;
        std::getline(status, rest_of_line);
    }
    return 0.0;
#endif
}

std::vector<int64_t> inferShapeForFlatInput(const std::size_t element_count) {
    return {1, static_cast<int64_t>(element_count)};
}

} // namespace edgeai::TensorUtils
