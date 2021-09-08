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

#pragma once

#include <popart/adam.hpp>
#include <popart/names.hpp>
#include <popart/optimizer.hpp>

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ipu {

class OptmizerMetaInfo {
 public:
  OptmizerMetaInfo();
  ~OptmizerMetaInfo();

  void SetType(const std::string &type) { type_ = type; }
  std::string GetType() const { return type_; }

  void SetAttr(const std::string &attr, float value);
  float GetAttr(const std::string &attr, float default_value = 0.0f) const;

  void SetLoss(const std::string &loss) { loss_ = loss; }
  std::string GetLoss() const { return loss_; }

  void SetLR(float lr_rate) { lr_rate_ = lr_rate; }
  float GetLR() const { return lr_rate_; }

  void SetLRVarName(const std::string &name) { lr_var_name_ = name; }
  std::string GetLRVarName() const { return lr_var_name_; }

 private:
  // type: adam, sgd, ...
  std::string type_;

  // loss: loss TensorId
  std::string loss_;

  // attrs: beta1, beta2, ...
  std::map<std::string, float> attrs_;

  // learning rate
  float lr_rate_;
  std::string lr_var_name_;
};

std::unique_ptr<popart::Optimizer> GetPopartOptimizer(
    const OptmizerMetaInfo &info);

}  // namespace ipu
}  // namespace framework
}  // namespace paddle
