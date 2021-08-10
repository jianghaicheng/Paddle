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

#include "paddle/fluid/framework/ipu/popart_canonicalization/canonicalization_utils.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace {

ir::Node *elementwise_add_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();
  auto op_desc = std::make_unique<framework::OpDesc>();
  op_desc->SetType("Add");

  std::vector<std::string> inputs;
  inputs.push_back(op->Input("X").front());
  inputs.push_back(op->Input("Y").front());
  op_desc->SetInput("__inputs__", inputs);
  std::vector<std::string> outputs;
  outputs.push_back(op->Output("Out").front());
  op_desc->SetOutput("__outputs__", outputs);

  op_desc->Flush();
  return graph->CreateOpNode(op_desc.get());
}

ir::Node *reduce_mean_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();
  auto op_desc = std::make_unique<framework::OpDesc>();
  op_desc->SetType("ReduceMean");

  std::vector<std::string> inputs;
  inputs.push_back(op->Input("X").front());
  op_desc->SetInput("__inputs__", inputs);
  std::vector<std::string> outputs;
  outputs.push_back(op->Output("Out").front());
  op_desc->SetOutput("__outputs__", outputs);
  auto reduce_all = BOOST_GET_CONST(bool, op->GetAttr("reduce_all"));
  if (!reduce_all) {
    auto axes_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("dim"));
    auto axes = std::vector<int64_t>{axes_.begin(), axes_.end()};
    op_desc->SetAttr("axes", axes);
  }
  auto keepdims_ = BOOST_GET_CONST(bool, op->GetAttr("keep_dim"));
  auto keepdims = int64_t{keepdims_};
  op_desc->SetAttr("keepdims", keepdims);

  op_desc->Flush();
  return graph->CreateOpNode(op_desc.get());
}

REGISTER_HANDLER(elementwise_add, elementwise_add_handler);
REGISTER_HANDLER(reduce_mean, reduce_mean_handler);

}  // namespace
}  // namespace framework
}  // namespace paddle
