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

ir::Node *fill_constant_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();
  auto op_desc = std::make_unique<framework::OpDesc>();
  op_desc->SetType("Constant");
  if (!op->Input("ShapeTensor").empty()) {
    PADDLE_THROW(
        platform::errors::Unimplemented("op fill_constant with ShapeTensor"));
  }
  std::vector<std::string> outputs;
  outputs.push_back(op->Output("Out").front());
  op_desc->SetOutput("__outputs__", outputs);

  auto dtype_ = BOOST_GET_CONST(int, op->GetAttr("dtype"));
  auto dtype = ConvertDataType(dtype_);
  op_desc->SetAttr("dtype", dtype);
  auto dims = BOOST_GET_CONST(std::vector<int64_t>, op->GetAttr("shape"));
  op_desc->SetAttr("dims", dims);
  auto value_ = BOOST_GET_CONST(float, op->GetAttr("value"));
  size_t size = 1;
  for (auto &dim : dims) {
    size *= dim;
  }
  Attribute value;
  switch (dtype_) {
    case proto::VarType::FP32:
      value = std::vector<float>(size, value_);
      break;
    case proto::VarType::FP64:
      value = std::vector<double>(size, value_);
      break;
    case proto::VarType::INT32:
      value = std::vector<int>(size, value_);
      break;
    case proto::VarType::INT64:
      value = std::vector<int64_t>(size, value_);
      break;
    default:
      PADDLE_THROW(
          platform::errors::Unimplemented("fill_constant dtype: %d", dtype_));
  }
  op_desc->SetAttr("value", value);

  op_desc->Flush();
  return graph->CreateOpNode(op_desc.get());
}

ir::Node *gaussian_random_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();
  auto op_desc = std::make_unique<framework::OpDesc>();
  op_desc->SetType("RandomNormal");

  std::vector<std::string> outputs;
  outputs.push_back(op->Output("Out").front());
  op_desc->SetOutput("__outputs__", outputs);

  auto shape = BOOST_GET_CONST(std::vector<int64_t>, op->GetAttr("shape"));
  op_desc->SetAttr("shape", shape);
  auto dtype_ = BOOST_GET_CONST(int, op->GetAttr("dtype"));
  auto dtype = ConvertDataType(dtype_);
  op_desc->SetAttr("dtype", dtype);

  auto mean = BOOST_GET_CONST(float, op->GetAttr("mean"));
  op_desc->SetAttr("mean", mean);
  auto std = BOOST_GET_CONST(float, op->GetAttr("std"));
  op_desc->SetAttr("scale", std);
  // seed TODO

  op_desc->Flush();
  return graph->CreateOpNode(op_desc.get());
}

ir::Node *uniform_random_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();
  auto op_desc = std::make_unique<framework::OpDesc>();
  op_desc->SetType("RandomUniform");

  std::vector<std::string> outputs;
  outputs.push_back(op->Output("Out").front());
  op_desc->SetOutput("__outputs__", outputs);

  auto shape = BOOST_GET_CONST(std::vector<int64_t>, op->GetAttr("shape"));
  op_desc->SetAttr("shape", shape);
  auto dtype_ = BOOST_GET_CONST(int, op->GetAttr("dtype"));
  auto dtype = ConvertDataType(dtype_);
  op_desc->SetAttr("dtype", dtype);
  auto max = BOOST_GET_CONST(float, op->GetAttr("max"));
  op_desc->SetAttr("high", max);
  auto min = BOOST_GET_CONST(float, op->GetAttr("min"));
  op_desc->SetAttr("low", min);
  // seed
  op_desc->Flush();
  return graph->CreateOpNode(op_desc.get());
}

ir::Node *transpose_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();

  auto axis_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("axis"));
  std::vector<int64_t> axis(axis_.begin(), axis_.end());
  auto attrs = AttributeMap{{"axis", axis}};

  auto new_node_transpose =
      CreateBaseOp(graph, "Transpose", node->inputs, node->outputs, attrs);
  ReplaceNodeOutputs(node, new_node_transpose);
  ReplaceNodeInputs(node, new_node_transpose);
  return new_node_transpose;
}

ir::Node *reshape_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();
  // TODO(yaozhixin) : Shape and ShapeTensor as inputs
  auto shape_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("shape"));
  std::vector<int64_t> shape(shape_.begin(), shape_.end());
  auto attrs = AttributeMap{
      {"value", shape},
      {"dims", std::vector<int64_t>{static_cast<int64_t>(shape.size())}},
      {"dtype", ONNXDataType::INT64}};
  auto new_node_const = CreateBaseOp(graph, "Constant", {}, {}, attrs);
  ReplaceNodeOutputs(node, new_node_const);

  auto new_node_reshape = CreateBaseOp(
      graph, "Reshape", {GetInputNode("X", node), new_node_const->outputs[0]},
      {GetOutputNode("Out", node)}, {});
  ReplaceNodeInputs(node, new_node_reshape);
  return new_node_reshape;
}

REGISTER_HANDLER(fill_constant, fill_constant_handler);
REGISTER_HANDLER(gaussian_random, gaussian_random_handler);
REGISTER_HANDLER(uniform_random, uniform_random_handler);
REGISTER_HANDLER(transpose2, transpose_handler);
REGISTER_HANDLER(reshape2, reshape_handler);

}  // namespace
}  // namespace ipu
}  // namespace framework
}  // namespace paddle
