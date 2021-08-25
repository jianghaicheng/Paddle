// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once
#include <popart/adam.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/names.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/optimizer.hpp>
#include <popart/session.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/stepio.hpp>
#include <popart/tensorinfo.hpp>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/ipu/ipu_strategy.h"
#include "paddle/fluid/framework/ipu/ipu_utils.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ipu {

class Compiler {
 public:
  explicit Compiler(const IpuStrategy *ipu_strategy);
  ~Compiler();
  void InitInputs(ir::Graph *graph, const std::vector<std::string> &feed_list);
  void InitOutputs(const std::vector<std::string> &fetch_list);
  void LowerWeights(const ir::Graph *graph, const Scope *scope_);
  void RegisterOpFunc();
  void LowerBody(const ir::Graph *graph);
  std::vector<std::string> GetOpInputs(const OpDesc *op);

  void InsertTensors(std::vector<std::string> output_names,
                     std::vector<std::string> tensor_ids);
  void InsertTensors(std::vector<std::string> output_names,
                     std::string tensor_id);

  void SetIpuIndexStage(const std::vector<std::string> &tensor_ids,
                        const OpDesc *op_desc);
  void SetIpuIndexStage(const std::string &tensor_id,
                        const OpDesc *op_desc);

  std::vector<popart::TensorId> GetInputs() { return inputs_; }
  std::vector<popart::TensorId> GetOutputs() { return outputs_; }
  std::map<std::string, popart::TensorId> GetTensors() { return tensors_; }
  std::vector<int64_t> GetTensorShape(std::string name) {
    return builder_->getTensorShape(tensors_[name]);
  }
  std::string GetModelProto() { return builder_->getModelProto(); };
  void SaveModelProto(std::string name) { builder_->saveModelProto(name); }

 private:
  std::map<std::string, popart::TensorId> tensors_;
  std::unique_ptr<popart::Builder> builder_;
  const IpuStrategy *ipu_strategy_;
  std::vector<popart::TensorId> inputs_;
  std::vector<popart::TensorId> outputs_;
  using Func = std::function<void(OpDesc *op_desc)>;
  std::unordered_map<std::string, Func> name_function_;
};

}  // namespace ipu
}  // namespace framework
}  // namespace paddle
