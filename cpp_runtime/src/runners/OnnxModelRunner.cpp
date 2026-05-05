#include "runners/OnnxModelRunner.hpp"

#include "core/TensorUtils.hpp"

#include <chrono>
#include <filesystem>
#include <numeric>
#include <stdexcept>

namespace edgeai {

#if EDGE_AI_ENABLE_ONNXRUNTIME

OnnxModelRunner::OnnxModelRunner()
    : env_(ORT_LOGGING_LEVEL_WARNING, "edge_ai_runtime") {
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
}

OnnxModelRunner::~OnnxModelRunner() = default;

void OnnxModelRunner::loadModel(const std::string& model_path) {
    if (!std::filesystem::exists(model_path)) {
        throw std::runtime_error("Model file does not exist: " + model_path);
    }

    model_path_ = model_path;
#ifdef _WIN32
    std::wstring wide_model_path(model_path.begin(), model_path.end());
    session_ = std::make_unique<Ort::Session>(env_, wide_model_path.c_str(), session_options_);
#else
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
#endif
    refreshModelMetadata();
    loaded_ = true;
}

void OnnxModelRunner::refreshModelMetadata() {
    Ort::AllocatorWithDefaultOptions allocator;
    input_names_storage_.clear();
    output_names_storage_.clear();
    input_names_.clear();
    output_names_.clear();

    const std::size_t input_count = session_->GetInputCount();
    const std::size_t output_count = session_->GetOutputCount();

    for (std::size_t i = 0; i < input_count; ++i) {
        auto name = session_->GetInputNameAllocated(i, allocator);
        input_names_storage_.emplace_back(name.get());
    }
    for (std::size_t i = 0; i < output_count; ++i) {
        auto name = session_->GetOutputNameAllocated(i, allocator);
        output_names_storage_.emplace_back(name.get());
    }
    for (const auto& name : input_names_storage_) {
        input_names_.push_back(name.c_str());
    }
    for (const auto& name : output_names_storage_) {
        output_names_.push_back(name.c_str());
    }

    if (input_count == 0 || output_count == 0) {
        throw std::runtime_error("ONNX model must expose at least one input and one output");
    }

    auto input_type_info = session_->GetInputTypeInfo(0);
    auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    input_shape_ = tensor_info.GetShape();
}

InferenceResult OnnxModelRunner::runInference(const std::string& job_id, const std::vector<float>& input_data) {
    if (!loaded_ || !session_) {
        throw std::runtime_error("ONNX model is not loaded");
    }
    if (input_data.empty()) {
        throw std::runtime_error("Input data is empty");
    }

    InferenceResult result;
    result.job_id = job_id;
    result.timestamp = TensorUtils::currentTimestampUtc();

    std::vector<int64_t> input_shape = input_shape_;
    int64_t known_product = 1;
    for (auto& dim : input_shape) {
        if (dim <= 0) {
            dim = 1;
        }
        known_product *= dim;
    }

    if (input_shape.empty() || known_product != static_cast<int64_t>(input_data.size())) {
        input_shape = TensorUtils::inferShapeForFlatInput(input_data.size());
    }

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(input_data.data()),
        input_data.size(),
        input_shape.data(),
        input_shape.size());

    const auto start = std::chrono::steady_clock::now();
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names_.data(),
        &input_tensor,
        1,
        output_names_.data(),
        output_names_.size());
    const auto end = std::chrono::steady_clock::now();

    result.inference_latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.success = true;

    if (!output_tensors.empty() && output_tensors.front().IsTensor()) {
        auto& output = output_tensors.front();
        auto tensor_info = output.GetTensorTypeAndShapeInfo();
        const auto element_count = tensor_info.GetElementCount();
        const float* values = output.GetTensorData<float>();
        result.output.assign(values, values + element_count);
    }

    return result;
}

#else

OnnxModelRunner::OnnxModelRunner() = default;
OnnxModelRunner::~OnnxModelRunner() = default;

void OnnxModelRunner::loadModel(const std::string& model_path) {
    if (!std::filesystem::exists(model_path)) {
        throw std::runtime_error("Model file does not exist: " + model_path);
    }
    model_path_ = model_path;
    throw std::runtime_error("ONNX Runtime backend is disabled. Reconfigure with EDGE_AI_ENABLE_ONNXRUNTIME=ON and ONNXRUNTIME_ROOT.");
}

InferenceResult OnnxModelRunner::runInference(const std::string& job_id, const std::vector<float>&) {
    InferenceResult result;
    result.job_id = job_id;
    result.timestamp = TensorUtils::currentTimestampUtc();
    result.success = false;
    result.error_message = "ONNX Runtime backend is disabled";
    return result;
}

#endif

std::string OnnxModelRunner::getModelName() const {
    return std::filesystem::path(model_path_).filename().string();
}

bool OnnxModelRunner::isLoaded() const {
    return loaded_;
}

std::unique_ptr<IModelRunner> createModelRunner(const std::string& runner_type) {
    if (runner_type == "onnx") {
        return std::make_unique<OnnxModelRunner>();
    }
    throw std::invalid_argument("Unsupported runner type: " + runner_type);
}

} // namespace edgeai
