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
#include "paddle/fluid/framework/ipu/popart_canonicalization/op_builder.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ipu {
namespace {

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

ir::Node *mean_handler(ir::Graph *graph, ir::Node *node) {
  auto new_node = CreateBaseOp(graph, "ReduceMean", {GetInputNode("X", node)},
                               {GetOutputNode("Out", node)},
                               {
                                   {"keepdims", int64_t{0}},
                               });
  ReplaceNodeInputs(node, new_node);
  ReplaceNodeOutputs(node, new_node);
  return new_node;
}

ir::Node *pow_handler(ir::Graph *graph, ir::Node *node) {
  // Op(pow) -> Op(Constant)->Var(const_out)->Op(Pow)
  auto *op = node->Op();
  auto value_ = BOOST_GET_CONST(float, op->GetAttr("factor"));
  auto attrs = MakeConstAttributeMap(value_, {1}, ONNXDataType::FLOAT);
  auto new_node_const = CreateConst(graph, {}, {}, attrs);
  auto new_node_pow = CreateBaseOp(
      graph, "Pow", {GetInputNode("X", node), new_node_const->outputs[0]},
      {node->outputs[0]});
  ReplaceNodeInputs(node, new_node_pow);
  return new_node_pow;
}

ir::Node *mul_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();
  auto op_desc = std::make_unique<framework::OpDesc>();
  op_desc->SetType("MatMul");

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

ir::Node *sum_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();
  auto op_desc = std::make_unique<framework::OpDesc>();
  op_desc->SetType("Sum");

  op_desc->SetInput("__inputs__", op->Input("X"));
  std::vector<std::string> outputs;
  outputs.push_back(op->Output("Out").front());
  op_desc->SetOutput("__outputs__", outputs);

  op_desc->Flush();
  return graph->CreateOpNode(op_desc.get());
}

ir::Node *softmax_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();
  auto op_desc = std::make_unique<framework::OpDesc>();
  op_desc->SetType("Softmax");

  std::vector<std::string> inputs;
  inputs.push_back(op->Input("X").front());
  op_desc->SetInput("__inputs__", inputs);
  std::vector<std::string> outputs;
  outputs.push_back(op->Output("Out").front());
  op_desc->SetOutput("__outputs__", outputs);

  auto axis_ = BOOST_GET_CONST(int, op->GetAttr("axis"));
  auto axis = int64_t{axis_};
  op_desc->SetAttr("axis", axis);

  op_desc->Flush();
  return graph->CreateOpNode(op_desc.get());
}

REGISTER_HANDLER(reduce_mean, reduce_mean_handler);
REGISTER_HANDLER(mean, mean_handler);
REGISTER_HANDLER(pow, pow_handler);
REGISTER_HANDLER(mul, mul_handler);
REGISTER_HANDLER(sum, sum_handler);
REGISTER_HANDLER(softmax, softmax_handler);

}  // namespace
}  // namespace ipu
}  // namespace framework
}  // namespace paddle
