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

#include "paddle/fluid/framework/ipu/popart_canonicalization_utils.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace {

ir::Node *conv2d_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();
  auto op_desc = std::make_unique<framework::OpDesc>();
  op_desc->SetType("Conv");

  std::vector<std::string> inputs;
  inputs.push_back(op->Input("Input").front());
  inputs.push_back(op->Input("Filter").front());
  if (op->HasInput("Bias")) {
    if (!op->Input("Bias").empty()) {
      inputs.push_back(op->Input("Bias").front());
    }
  }
  op_desc->SetInput("__inputs__", inputs);
  std::vector<std::string> outputs;
  outputs.push_back(op->Output("Output").front());
  op_desc->SetOutput("__outputs__", outputs);

  auto dilations_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("dilations"));
  auto dilations = std::vector<int64_t>{dilations_.begin(), dilations_.end()};
  auto group_ = BOOST_GET_CONST(int, op->GetAttr("groups"));
  auto group = int64_t{group_};
  // auto paddings_ = BOOST_GET_CONST(std::vector<int>,
  // op->GetAttr("paddings")); auto pads =
  // std::vector<int64_t>{paddings_.begin(), paddings_.end()};
  auto pads = std::vector<int64_t>{1, 1, 1, 1};
  auto stride_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("strides"));
  auto stride = std::vector<int64_t>{stride_.begin(), stride_.end()};
  op_desc->SetAttr("dilations", dilations);
  op_desc->SetAttr("group", group);
  op_desc->SetAttr("pads", pads);
  op_desc->SetAttr("strides", stride);

  op_desc->Flush();
  return graph->CreateOpNode(op_desc.get());
}

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
  if (reduce_all) {
    // TODO(alleng) get axes from input tensor shape/dim
    auto axes = std::vector<int64_t>{0, 1, 2, 3};
    op_desc->SetAttr("axes", axes);
  } else {
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

ir::Node *uniform_random_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();
  auto op_desc = std::make_unique<framework::OpDesc>();
  op_desc->SetType("RandomUniform");

  std::vector<std::string> outputs;
  outputs.push_back(op->Output("Out").front());
  op_desc->SetOutput("__outputs__", outputs);

  auto shape = BOOST_GET_CONST(std::vector<int64_t>, op->GetAttr("shape"));
  op_desc->SetAttr("shape", shape);
  // auto dtype = BOOST_GET_CONST(int, op->GetAttr("dtype"));
  op_desc->SetAttr("dtype", 1);
  // cvt dtype
  /*
    enum Type {
    // Pod Types
    BOOL = 0;
    INT16 = 1;
    INT32 = 2;
    INT64 = 3;
    FP16 = 4;
    FP32 = 5;
    FP64 = 6;
    // Tensor<size_t> is used in C++.
    SIZE_T = 19;
    UINT8 = 20;
    INT8 = 21;
    BF16 = 22;
    COMPLEX64 = 23;
    COMPLEX128 = 24;
    ...
  */
  auto max = BOOST_GET_CONST(float, op->GetAttr("max"));
  op_desc->SetAttr("high", max);
  auto min = BOOST_GET_CONST(float, op->GetAttr("min"));
  op_desc->SetAttr("low", min);
  // seed
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
  // auto dtype = BOOST_GET_CONST(int, op->GetAttr("dtype"));
  op_desc->SetAttr("dtype", 1);

  auto mean = BOOST_GET_CONST(float, op->GetAttr("mean"));
  op_desc->SetAttr("mean", mean);
  auto std = BOOST_GET_CONST(float, op->GetAttr("std"));
  op_desc->SetAttr("scale", std);
  // seed TODO

  op_desc->Flush();
  return graph->CreateOpNode(op_desc.get());
}

ir::Node *fill_constant_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();
  auto op_desc = std::make_unique<framework::OpDesc>();
  op_desc->SetType("ConstantOfShape");

  std::vector<std::string> outputs;
  outputs.push_back(op->Output("Out").front());
  op_desc->SetOutput("__outputs__", outputs);

  auto shape = BOOST_GET_CONST(std::vector<int64_t>, op->GetAttr("shape"));
  op_desc->SetAttr("shape", shape);
  // auto dtype = BOOST_GET_CONST(int, op->GetAttr("dtype"));
  op_desc->SetAttr("dtype", 1);

  auto value = BOOST_GET_CONST(float, op->GetAttr("value"));
  op_desc->SetAttr("value", value);

  op_desc->Flush();
  return graph->CreateOpNode(op_desc.get());
}

REGISTER_HANDLER(conv2d, conv2d_handler);
REGISTER_HANDLER(elementwise_add, elementwise_add_handler);
REGISTER_HANDLER(reduce_mean, reduce_mean_handler);
REGISTER_HANDLER(uniform_random, uniform_random_handler);
REGISTER_HANDLER(gaussian_random, gaussian_random_handler);
REGISTER_HANDLER(fill_constant, fill_constant_handler);

}  // namespace
}  // namespace framework
}  // namespace paddle
