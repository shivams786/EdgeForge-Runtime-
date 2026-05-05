#pragma once

#include "core/InferenceResult.hpp"

#include <memory>
#include <string>
#include <vector>

namespace edgeai {

class IModelRunner {
public:
    virtual ~IModelRunner() = default;
    virtual void loadModel(const std::string& model_path) = 0;
    virtual InferenceResult runInference(const std::string& job_id, const std::vector<float>& input_data) = 0;
    virtual std::string getModelName() const = 0;
    virtual bool isLoaded() const = 0;
};

} // namespace edgeai
