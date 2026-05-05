#include "test_common.hpp"

#include "core/TensorUtils.hpp"

#include <filesystem>
#include <fstream>

EDGEAI_TEST(tensor_utils_parses_text_csv_and_json_array) {
    const auto text_values = edgeai::TensorUtils::parseFloatVectorText("0.1 0.2\n0.3");
    edgeai::test::require(text_values.size() == 3, "text parse size mismatch");

    const auto csv_values = edgeai::TensorUtils::parseFloatVectorText("1.0,2.0,3.5");
    edgeai::test::require(csv_values.size() == 3, "csv parse size mismatch");
    edgeai::test::require(csv_values[2] == 3.5F, "csv parse value mismatch");

    const auto json_values = edgeai::TensorUtils::parseFloatVectorText("[4.0, 5.0, 6.0]");
    edgeai::test::require(json_values.size() == 3, "json parse size mismatch");
}

EDGEAI_TEST(tensor_utils_reads_file) {
    const auto path = std::filesystem::temp_directory_path() / "edge_ai_runtime_tensor_test.txt";
    {
        std::ofstream out(path);
        out << "0.1 0.2 0.3 0.4";
    }

    const auto values = edgeai::TensorUtils::readFloatVector(path);
    edgeai::test::require(values.size() == 4, "file parse size mismatch");

    std::filesystem::remove(path);
}

