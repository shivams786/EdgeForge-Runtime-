#pragma once

#include "core/IModelRunner.hpp"

#include <memory>
#include <string>

namespace edgeai {

std::unique_ptr<IModelRunner> createModelRunner(const std::string& runner_type);

} // namespace edgeai

