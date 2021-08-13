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

#include "paddle/fluid/framework/ipu/popart_canonicalization/op_builder.h"

#include "paddle/fluid/framework/ipu/popart_canonicalization/canonicalization_utils.h"

namespace paddle {
namespace framework {
namespace ipu {

// singleton
static int var_count = 0;

std::string GenerateVarName() {
  return std::string("_popart_gen_") + std::to_string(var_count++);
}

ir::Node *MakeVarNode(ir::Graph *graph) {
  auto var_name = GenerateVarName();
  auto var_desc = std::make_unique<framework::VarDesc>(var_name);

  auto var = graph->CreateVarNode(var_desc.get());
  return var;
}

ir::Node *MakeOpNode(ir::Graph *graph, const std::string &type,
                     const std::vector<ir::Node *> &inputs,
                     const std::vector<ir::Node *> &outputs) {
  auto op_desc = std::make_unique<framework::OpDesc>();
  op_desc->SetType(type);
  auto op = graph->CreateOpNode(op_desc.get());

  for (auto *in : inputs) {
    ConnectNodes(in, op);
  }
  if (outputs.empty()) {
    auto var = MakeVarNode(graph);
    ConnectNodes(op, var);
  } else {
    for (auto *out : outputs) {
      ConnectNodes(op, out);
    }
  }

  // i/o
  std::vector<std::string> input_names;
  for (auto node : op->inputs) {
    input_names.push_back(node->Name());
  }
  op->Op()->SetInput("__inputs__", input_names);
  std::vector<std::string> output_names;
  for (auto node : op->outputs) {
    output_names.push_back(node->Name());
  }
  op->Op()->SetOutput("__outputs__", output_names);
  op->Op()->Flush();

  return op;
}

Node *CreateBaseOp(ir::Graph *graph, const std::string &type,
                   const std::vector<ir::Node *> &inputs,
                   const std::vector<ir::Node *> &outputs,
                   const AttributeMap &attrs) {
  auto node = MakeOpNode(graph, type, inputs, outputs);
  if (!attrs.empty()) {
    node->Op()->SetAttrMap(attrs);
  }
  return node;
}

AttributeMap MakeConstAttributeMap(float v, std::vector<int64_t> dims,
                                   int dtype) {
  size_t size = 1;
  for (auto &dim : dims) {
    size *= dim;
  }
  std::vector<float> value(size, v);
  return AttributeMap{
      {"value", value}, {"dims", dims}, {"dtype", dtype},
  };
}

ir::Node *CreateConst(ir::Graph *graph, const std::vector<ir::Node *> &inputs,
                      const std::vector<ir::Node *> &outputs,
                      const AttributeMap &attrs) {
  return CreateBaseOp(graph, "Constant", inputs, outputs, attrs);
}

}  // namespace ipu
}  // namespace framework
}  // namespace paddle
