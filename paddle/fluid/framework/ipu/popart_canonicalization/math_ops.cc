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
namespace ipu {
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

ir::Node *pow_handler(ir::Graph *graph, ir::Node *node) {
  // Op(pow) -> Op(Constant)->Var(const_out)->Op(Pow)
  auto *op = node->Op();
  // Op(Constant)
  auto op_const = std::make_unique<framework::OpDesc>();
  op_const->SetType("Constant");
  std::string op_const_out = op->Output("Out").front() + ":__0_";
  auto value_ = BOOST_GET_CONST(float, op->GetAttr("factor"));
  auto value = std::vector<float>{value_};
  op_const->SetAttr("value", value);
  auto dims = std::vector<int64_t>{1};
  op_const->SetAttr("dims", dims);
  op_const->SetAttr("dtype", ONNXDataType::FLOAT);
  std::vector<std::string> outputs_const;
  outputs_const.push_back(op_const_out);
  op_const->SetOutput("__outputs__", outputs_const);
  op_const->Flush();
  // Var(const_out)
  auto var_const = std::make_unique<framework::VarDesc>(op_const_out);
  var_const->SetType(proto::VarType::LOD_TENSOR);
  var_const->SetDataType(proto::VarType::FP32);
  auto shape_var_const = std::vector<int64_t>{1};
  var_const->SetShape(shape_var_const);
  auto var_node_const = graph->CreateVarNode(var_const.get());
  auto node_const = graph->CreateOpNode(op_const.get());
  MoveNodeInputs(node, node_const);
  ConnectNodes(node_const, var_node_const);
  // Op(Pow)
  auto op_pow = std::make_unique<framework::OpDesc>();
  op_pow->SetType("Pow");
  std::vector<std::string> inputs;
  inputs.push_back(op->Input("X").front());
  inputs.push_back(op_const->Output("__outputs__").front());
  op_pow->SetInput("__inputs__", inputs);
  std::vector<std::string> outputs;
  outputs.push_back(op->Output("Out").front());
  op_pow->SetOutput("__outputs__", outputs);
  op_pow->Flush();
  auto node_pow = graph->CreateOpNode(op_pow.get());
  ConnectNodes(var_node_const, node_pow);
  MoveNodeOutputs(node, node_pow);
  return node_pow;
}

REGISTER_HANDLER(elementwise_add, elementwise_add_handler);
REGISTER_HANDLER(reduce_mean, reduce_mean_handler);
REGISTER_HANDLER(pow, pow_handler);

}  // namespace
}  // namespace ipu
}  // namespace framework
}  // namespace paddle
