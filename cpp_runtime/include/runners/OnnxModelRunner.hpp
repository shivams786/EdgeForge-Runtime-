#pragma once

#include "core/IModelRunner.hpp"

#include <memory>
#include <string>
#include <vector>

#if EDGE_AI_ENABLE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace edgeai {

class OnnxModelRunner final : public IModelRunner {
public:
    OnnxModelRunner();
    ~OnnxModelRunner() override;

    void loadModel(const std::string& model_path) override;
    InferenceResult runInference(const std::string& job_id, const std::vector<float>& input_data) override;
    std::string getModelName() const override;
    bool isLoaded() const override;

private:
    std::string model_path_;
    bool loaded_{false};

#if EDGE_AI_ENABLE_ONNXRUNTIME
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_names_storage_;
    std::vector<std::string> output_names_storage_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<int64_t> input_shape_;

    void refreshModelMetadata();
#endif
};

} // namespace edgeai

