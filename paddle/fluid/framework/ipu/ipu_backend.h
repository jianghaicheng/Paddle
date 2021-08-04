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

#include <map>
#include <string>
#include <vector>

#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/names.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/session.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/stepio.hpp>
#include <popart/tensorinfo.hpp>

#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/ipu/ipu_build_strategy.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

using ipu::IpuBuildStrategy;

struct Optimizer {
  std::string type_;
  std::string loss_;
  // as far as we know, attr is usually float
  std::map<std::string, float> attrs_;
};

class IpuBackend {
 public:
  IpuBackend();

  void Compile(ir::Graph *graph, const std::vector<std::string> &feed_list,
               const std::vector<std::string> &fetch_list);

  void Run(const std::vector<const Tensor *> &inputs,
           const std::vector<Tensor *> &outputs);

  std::string GetOptimizerType() { return optimizer_.type_; }

  void SetOptimizerType(const std::string &type) { optimizer_.type_ = type; }

  const std::map<std::string, float> &GetOptimizerAttr() {
    return optimizer_.attrs_;
  }

  void SetOptimizerAttr(const std::string &attr, float value) {
    optimizer_.attrs_[attr] = value;
  }

  void SetLoss(const std::string &loss) { optimizer_.loss_ = loss; }

  std::vector<int64_t> GetTensorShape(const std::string &var_name) {
    return builder_->getTensorShape(tensors_[var_name]);
  }

  // SetScope, so we can get model parameters from scope
  void SetScope(Scope *scope) { scope_ = scope; }

  static std::shared_ptr<IpuBackend> GetInstance() {
    if (NULL == instance_) {
      instance_.reset(new IpuBackend());
    }
    return instance_;
  }

 private:
  void Prepare();
  void LowerWeights(const ir::Graph *);
  void LowerBody(const ir::Graph *);
  std::vector<std::string> GetOpInputs(const OpDesc *op);

 private:
  Optimizer optimizer_;
  IpuBuildStrategy ipu_build_strategy_;
  Scope *scope_ = nullptr;

  std::vector<popart::TensorId> inputs_;
  std::vector<popart::TensorId> outputs_;
  std::map<std::string, popart::TensorId> tensors_;

  std::unique_ptr<popart::Builder> builder_;
  std::unique_ptr<popart::Session> session_;

  static std::shared_ptr<IpuBackend> instance_;
};

}  // namespace framework
}  // namespace paddle
