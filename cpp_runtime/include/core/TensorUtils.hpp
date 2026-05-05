#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace edgeai::TensorUtils {

std::vector<float> readFloatVector(const std::filesystem::path& input_path);
std::vector<float> parseFloatVectorText(const std::string& content);
std::string currentTimestampUtc();
std::string jsonEscape(const std::string& value);
double getCurrentMemoryUsageMb();
std::vector<int64_t> inferShapeForFlatInput(std::size_t element_count);

} // namespace edgeai::TensorUtils

