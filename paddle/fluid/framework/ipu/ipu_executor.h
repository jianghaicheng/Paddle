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

#include <popart/dataflow.hpp>
#include <popart/half.hpp>
#include <popart/names.hpp>
#include <popart/session.hpp>
#include <popart/tensorinfo.hpp>

#include "paddle/fluid/framework/ipu/common.h"
#include "paddle/fluid/framework/ipu/ipu_optimizer.h"
#include "paddle/fluid/framework/ipu/ipu_strategy.h"
#include "paddle/fluid/framework/ipu/ipu_utils.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {
namespace ipu {

using float16 = paddle::platform::float16;
class Executor {
 public:
  Executor();
  ~Executor();
  void Prepare(const std::string &proto,
               const std::map<std::string, popart::TensorId> &tensors,
               const std::vector<popart::TensorId> &outputs,
               std::shared_ptr<popart::DeviceInfo> device);

  void Run(const std::vector<popart::TensorId> &inputs_id,
           const std::vector<const Tensor *> &inputs,
           const std::vector<popart::TensorId> &outputs_id,
           const std::vector<Tensor *> &outputs,
           const framework::ExecutionContext &ctx);

  // Optimizer
  void SetOptimizerType(const std::string &type);
  void SetOptimizerAttr(const std::string &attr, float value);
  void SetOptimizerDType(popart::DataType type);
  void SetLoss(const std::string &loss);
  void SetLR(float lr_rate);
  void SetLRVarName(const std::string &name);

  void SetWeights(const std::vector<popart::TensorId> &info);

  void SetWeightsIO();
  void ConvertWeights(bool align_to_popart);
  void WeightsFromPaddle();
  void WeightsToPaddle();

  // Scope
  void SetScope(const Scope *scope) { scope_ = scope; }

  // Strategy
  void SetIpuStrategy(const IpuStrategy &strategy);

 private:
  float GetLRFromScope();

 public:
  OptmizerMetaInfo opt_info;
  std::unique_ptr<popart::Session> session_;

 private:
  const Scope *scope_ = nullptr;
  const IpuStrategy *ipu_strategy_ = nullptr;
  // weights_io_: map<tensor_id, paddle_var_ptr>
  popart::WeightsIO weights_io_;
  // Just include weights, exclude optimizer states
  std::vector<popart::TensorId> weights_;
  // <popart_var, paddle_var> pairs, include weights and optimizer states
  std::vector<std::pair<popart::TensorId, popart::TensorId>>
      weights_and_opt_state_;
  int step_ = 0;
};

}  // namespace ipu
}  // namespace framework
}  // namespace paddle
