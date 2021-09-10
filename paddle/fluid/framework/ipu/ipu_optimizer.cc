/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ipu/ipu_optimizer.h"

namespace paddle {
namespace framework {
namespace ipu {

OptmizerMetaInfo::OptmizerMetaInfo() {}

OptmizerMetaInfo::~OptmizerMetaInfo() {}

float OptmizerMetaInfo::GetAttr(const std::string &attr,
                                float default_value) const {
  if (attrs_.count(attr) == 0) {
    return default_value;
  }
  return attrs_.at(attr);
}

void OptmizerMetaInfo::SetAttr(const std::string &attr, float value) {
  attrs_[attr] = value;
}

std::unique_ptr<popart::Optimizer> GetPopartOptimizer(
    const OptmizerMetaInfo &opt_meta_info) {
  auto opt_type = opt_meta_info.GetType();
  PADDLE_ENFORCE_NE(opt_type, "", platform::errors::InvalidArgument(
                                      "Optimizer type have not been set."));

  if (opt_type == "sgd") {
    auto optimizer = std::make_unique<popart::SGD>(
        popart::OptimizerValue(opt_meta_info.GetLR(), false),
        popart::OptimizerValue(popart::SGD::getUnsetWeightDecay()),
        popart::OptimizerValue(popart::SGD::getUnsetMomentum()),
        popart::OptimizerValue(popart::SGD::getUnsetDampening()),
        popart::OptimizerValue(popart::SGD::getUnsetVelocityScaling()),
        popart::OptimizerValue(popart::SGD::getUnsetLossScaling()));
    return optimizer;
  } else if (opt_type == "adam") {
    auto optimizer = std::make_unique<popart::Adam>(
        popart::OptimizerValue(opt_meta_info.GetLR(), false),
        popart::OptimizerValue(popart::Adam::getUnsetWeightDecay()),
        popart::OptimizerValue(opt_meta_info.GetAttr("beta1"), false),
        popart::OptimizerValue(opt_meta_info.GetAttr("beta2"), false),
        popart::OptimizerValue(opt_meta_info.GetAttr("epsilon"), false),
        popart::OptimizerValue(popart::Adam::getUnsetLossScaling()),
        popart::AdamMode::Adam, popart::WeightDecayMode::Decay,
        popart::DataType::FLOAT, popart::DataType::FLOAT,
        popart::DataType::FLOAT);
    return optimizer;
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Optimizer %s is not implemented now.", opt_type));
  }
}

std::vector<std::pair<std::string, std::string>>
GetOptPrePostfix(const std::string& opt_type) {
  // format: {popart_tensor_id, paddle_tensor_id}, ...
  std::vector<std::pair<std::string, std::string>> pre_post_fix;

  pre_post_fix.push_back(std::make_pair("", ""));
  if (opt_type == "sgd") {
  } else if (opt_type == "adam") {
    pre_post_fix.push_back(std::make_pair("Accl1___", "_moment1_0"));
    pre_post_fix.push_back(std::make_pair("Accl2___", "_moment2_0"));
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Optimizer %s is not implemented now.", opt_type));
  }
  return pre_post_fix;
}

}  // namespace ipu
}  // namespace framework
}  // namespace paddle
