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

#include "paddle/fluid/framework/ir/ipu/ipu_inplace_pass.h"

#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {

Node *GetInputVarNode(const std::string &var_name, const ir::Node *node) {
  PADDLE_ENFORCE_EQ(node->IsOp(), true,
                    platform::errors::InvalidArgument("node is not Op"));
  for (auto *n : node->inputs) {
    if (n->Name() == var_name) {
      return n;
    }
  }
  return nullptr;
}

void IpuInplacePass::ApplyImpl(ir::Graph *graph) const {
  // use this pass after forward_graph_extract_pass
  VLOG(10) << "enter IpuInplacePass::ApplyImpl";
  VLOG(10) << "Raw Graph: ";
  VLOG(10) << DebugString(graph);

  for (auto *node : graph->Nodes()) {
    if (!node->IsOp()) {
      continue;
    }

    RenameInplaceVar(node);
  }

  VLOG(10) << "Post Graph: ";
  VLOG(10) << DebugString(graph);
  VLOG(10) << "leave IpuInplacePass::ApplyImpl";
}

void IpuInplacePass::RenameInplaceVar(ir::Node *node) const {
  // rename input_var, only support one input_var rename
  auto *op = node->Op();
  for (auto name : op->Input("__inputs__")) {
    for (auto name_out : op->Output("__outputs__")) {
      if (name == name_out) {
        auto new_name = name + "_1";
        VLOG(10) << "replace op node: " << node->Name()
                 << " input var: " << name << " to " << new_name;
        auto var = GetInputVarNode(name, node);
        if (var) {
          var->RenameVar(new_name);
          for (auto *op_in : var->inputs) {
            op_in->Op()->RenameOutput(name, new_name);
          }
          for (auto *op_out : var->outputs) {
            op_out->Op()->RenameInput(name, new_name);
          }
          return;
        }
      }
    }
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(ipu_inplace_pass, paddle::framework::ir::IpuInplacePass);
