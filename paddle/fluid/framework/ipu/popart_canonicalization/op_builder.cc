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

ir::Node *MakeVarNode(ir::Graph *graph, ir::Node *node) {
  auto var_name = GenerateVarName();
  auto var_desc = std::make_unique<framework::VarDesc>(var_name);

  auto var = graph->CreateVarNode(var_desc.get());
  return var;
}

ir::Node *MakeOpNode(ir::Graph *graph, ir::Node *node, const std::string &type,
                     const std::vector<ir::Node *> &inputs,
                     const std::vector<ir::Node *> &outputs) {
  auto op_desc = std::make_unique<framework::OpDesc>();
  op_desc->SetType(type);
  auto op = graph->CreateOpNode(op_desc.get());

  for (auto *in : inputs) {
    ConnectNodes(in, op);
  }
  if (outputs.empty()) {
    auto var = MakeVarNode(graph, node);
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

ir::Node *CreateBaseOp(ir::Graph *graph, ir::Node *node,
                       const std::string &type,
                       const std::vector<ir::Node *> &inputs,
                       const std::vector<ir::Node *> &outputs,
                       const AttributeMap &attrs) {
  auto new_node = MakeOpNode(graph, node, type, inputs, outputs);
  if (!attrs.empty()) {
    new_node->Op()->SetAttrMap(attrs);
  }
  // deal special attr
  if (!new_node->Op()->HasAttr("ipu_index")) {
    CopyOpAttr("ipu_index", node->Op(), new_node->Op());
  }
  if (!new_node->Op()->HasAttr("ipu_stage")) {
    CopyOpAttr("ipu_stage", node->Op(), new_node->Op());
  }

  return new_node;
}

ir::Node *CreateConst(ir::Graph *graph, ir::Node *node,
                      const std::vector<ir::Node *> &inputs,
                      const std::vector<ir::Node *> &outputs,
                      const AttributeMap &attrs) {
  return CreateBaseOp(graph, node, "popart_constant", inputs, outputs, attrs);
}

ir::Node *CreateCast(ir::Graph *graph, ir::Node *node,
                     const std::vector<ir::Node *> &inputs,
                     const std::vector<ir::Node *> &outputs, const int otype) {
  auto to = VarType2PopStr(otype);
  return CreateBaseOp(graph, node, "popart_cast", inputs, outputs,
                      {{"to", to}});
}

ir::Node *CreateGemm(ir::Graph *graph, ir::Node *node,
                     const std::vector<ir::Node *> &inputs,
                     const std::vector<ir::Node *> &outputs, int64_t transA,
                     int64_t transB, float alpha, float beta) {
  return CreateBaseOp(graph, node, "popart_gemm", inputs, outputs,
                      {
                          {"alpha", alpha},
                          {"beta", beta},
                          {"transA", transA},
                          {"transB", transB},
                      });
}

ir::Node *CreateReshape(ir::Graph *graph, ir::Node *node,
                        const std::vector<ir::Node *> &inputs,
                        const std::vector<ir::Node *> &outputs,
                        const std::vector<int64_t> &oshape) {
  auto attr = AttributeMap{
      {"value", oshape},
      {"dims", std::vector<int64_t>{static_cast<int64_t>(oshape.size())}},
      {"dtype", ONNXDataType::INT64}};
  auto new_node_const =
      CreateBaseOp(graph, node, "popart_constant", {}, {}, attr);
  auto new_node_reshape =
      CreateBaseOp(graph, node, "popart_reshape",
                   {inputs[0], new_node_const->outputs[0]}, outputs);
  return new_node_reshape;
}

ir::Node *CreateConv(ir::Graph *graph, ir::Node *node,
                     const std::vector<ir::Node *> &inputs,
                     const std::vector<ir::Node *> &outputs,
                     const std::vector<int64_t> &dilations, int64_t group,
                     const std::vector<int64_t> &kernel_shape,
                     const std::vector<int64_t> &pads,
                     const std::vector<int64_t> &strides) {
  auto attrs = AttributeMap{
      {"dilations", dilations},       {"group", group},
      {"kernel_shape", kernel_shape}, {"pads", pads},
      {"strides", strides},
  };
  return CreateBaseOp(graph, node, "popart_conv", inputs, outputs, attrs);
}

}  // namespace ipu
}  // namespace framework
}  // namespace paddle
