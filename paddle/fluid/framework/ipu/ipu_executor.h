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
#include <popart/names.hpp>
#include <popart/session.hpp>

#include "paddle/fluid/framework/ipu/common.h"
#include "paddle/fluid/framework/ipu/ipu_optimizer.h"
#include "paddle/fluid/framework/ipu/ipu_strategy.h"
#include "paddle/fluid/framework/ipu/ipu_utils.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {
namespace ipu {

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
           const std::vector<Tensor *> &outputs);

  // Optimizer
  void SetOptimizerType(const std::string &type);
  void SetOptimizerAttr(const std::string &attr, float value);
  void SetLoss(const std::string &loss);
  void SetLR(float lr_rate);
  void SetLRVarName(const std::string &name);

  void SetWeightsInfo(const std::vector<IdToInfo> &info);
  void UpdateHostOptimizer();

  // Scope
  void SetScope(Scope *scope) { scope_ = scope; }

  // Strategy
  void SetIpuStrategy(const IpuStrategy &strategy);

  // Outputs
  void SetOutputsShape(const std::map<std::string, std::vector<int64_t>> &info);
  std::vector<int64_t> GetOutputShape(const std::string &fetch_name);

 private:
  float GetLRFromScope();

 public:
  OptmizerMetaInfo opt_info;
  std::unique_ptr<popart::Session> session_;

 private:
  Scope *scope_ = nullptr;
  const IpuStrategy *ipu_strategy_ = nullptr;
  std::vector<IdToInfo> weights_info_;
  std::map<std::string, std::vector<int64_t>> outputs_shape_;
};

}  // namespace paddle
}  // namespace framework
}  // namespace ipu
