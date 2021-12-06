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

#include <popart/builder.hpp>
#include <popart/graphtransformer.hpp>
#include <popart/optimizer.hpp>
#include "paddle/fluid/framework/ipu/common.h"
#include "paddle/fluid/framework/ipu/ipu_strategy.h"
#include "paddle/fluid/framework/ipu/ipu_utils.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {
namespace ipu {

struct SharedObj {
  // TODO(alleng) better name, better organization

  // inputs_ & outputs_ save popart tensor id
  std::vector<popart::TensorId> inputs_;
  std::vector<popart::TensorId> outputs_;

  // <paddle_var, popart_var>
  std::map<std::string, popart::TensorId> tensors_;

  std::vector<popart::TensorId> weights_;

  std::string loss_var;
  std::string lr_var;
  float lr;

  using OptimizerFn =
      std::function<std::unique_ptr<popart::Optimizer>(float lr)>;
  OptimizerFn optimizer_fn;
  std::string optimizer_type;

  bool with_lr_sched = false;

 public:
  popart::Optimizer *Optimizer() { return optimizer.get(); }

  popart::Optimizer *NewOptimizer() {
    optimizer = optimizer_fn(lr);
    return optimizer.get();
  }

  popart::Optimizer *UpdateOptimizer(float lr_new) {
    optimizer = optimizer_fn(lr_new);
    return optimizer.get();
  }

 private:
  std::unique_ptr<popart::Optimizer> optimizer;
};

class Compiler {
 public:
  Compiler();
  void RegisterOpFunc();
  void LowerBody(const ir::Graph *graph);
  void InitInputs(ir::Graph *graph, const std::vector<std::string> &feed_list);
  void InitOutputs(const std::vector<std::string> &fetch_list);
  void LowerConstants(const ir::Graph *graph, const Scope *scope);
  void LowerWeights(const ir::Graph *graph, const Scope *scope);
  void LowerOptimier(const ir::Graph *graph, const Scope *scope);

  void InsertTensors(const std::vector<std::string> &output_names,
                     const std::vector<std::string> &tensor_ids);
  void InsertTensors(const std::vector<std::string> &output_names,
                     const std::string &tensor_id);
  void SetIpuIndexStage(const std::vector<std::string> &tensor_ids,
                        const OpDesc *op_desc);
  void SetIpuIndexStage(const std::string &tensor_id, const OpDesc *op_desc);
  void SetAMPAttributes(const std::vector<std::string> &tensor_ids,
                        const OpDesc *op_desc);
  void SetAMPAttributes(const std::string &tensor_id, const OpDesc *op_desc);
  void SetSerializeAttributes(const std::vector<std::string> &tensor_ids,
                              const OpDesc *op_desc);
  void SetSerializeAttributes(const std::string &tensor_id,
                              const OpDesc *op_desc);

  void SetIpuStrategy(const IpuStrategy &strategy) {
    ipu_strategy_ = &strategy;
  }

  void SetCustomOps(const std::vector<IpuCustomOpIdentifier> &custom_ops);

  std::string GetModelProto();
  void SaveModelProto(const std::string &path);
  void SaveModelProtoNoCheck(const std::string &path);
  void ConvertProtoToFp16();

  std::unique_ptr<SharedObj> shared_obj;

 private:
  std::vector<std::string> GetOpInputs(const OpDesc *op);
  const std::vector<std::string> &GetOpOutputs(const OpDesc *op);
  popart::DebugContext BuildDebugContext(const OpDesc *op);

 private:
  std::unique_ptr<popart::Builder> builder_;

  using OpFunc = std::function<void(OpDesc *op_desc)>;
  std::unordered_map<std::string, OpFunc> name_function_;

  // feed_list_ & fetch_list save paddle tensor id
  std::vector<std::string> feed_list_;
  std::vector<std::string> fetch_list_;

  std::string converted_proto_ = "";
  const IpuStrategy *ipu_strategy_ = nullptr;
  std::map<std::string, IpuCustomOpIdentifier> custom_ops_;
};

}  // namespace ipu
}  // namespace framework
}  // namespace paddle
