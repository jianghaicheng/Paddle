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

#include "paddle/fluid/framework/ir/ipu/popart_canonicalization_pass.h"

#include "paddle/fluid/framework/ipu/popart_canonicalization_utils.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {

void PopartCanonicalizationPass::ApplyImpl(ir::Graph* graph) const {
  // register op_handlers
  // call op_handler, get inputs and attrs
  // create new node in paddle, map attr to popart, mark raw node to remove
  // remove paddle raw node

  VLOG(10) << "enter PopartCanonicalizationPass::ApplyImpl";
  VLOG(10) << "Raw Graph: ";
  VLOG(10) << DebugString(graph);

  auto nodes = graph->Nodes();
  for (auto* node : nodes) {
    if (!node->IsOp()) {
      continue;
    }
    auto* op = node->Op();
    auto op_type = op->Type();

    ir::Node* new_node = nullptr;
    SymbolHandler handler = GetHandler(op_type);
    if (handler) {
      new_node = handler(graph, node);
      new_node->inputs = node->inputs;
      new_node->outputs = node->outputs;
      // restore node releations
      for (auto* node_in : node->inputs) {
        for (size_t i = 0; i < node_in->outputs.size(); ++i) {
          if (node_in->outputs[i] == node) {
            node_in->outputs[i] = new_node;
            break;
          }
        }
      }
      for (auto* node_out : node->outputs) {
        for (size_t i = 0; i < node_out->inputs.size(); ++i) {
          if (node_out->inputs[i] == node) {
            node_out->inputs[i] = new_node;
            break;
          }
        }
      }
    }
    if (new_node) {
      graph->RemoveNode(node);
    }
  }
  VLOG(10) << "Post Graph: ";
  VLOG(10) << DebugString(graph);
  VLOG(10) << "leave PopartCanonicalizationPass::ApplyImpl";
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(popart_canonicalization_pass,
              paddle::framework::ir::PopartCanonicalizationPass);
