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

Node *GetInVarNode(const std::string &var_name, const ir::Node *node) {
  PADDLE_ENFORCE_EQ(node->IsOp(), true,
                    platform::errors::InvalidArgument("node is not Op"));
  for (auto *n : node->inputs) {
    if (n->Name() == var_name) {
      return n;
    }
  }
  return nullptr;
}

Node *GetOutVarNode(const std::string &var_name, const ir::Node *node) {
  PADDLE_ENFORCE_EQ(node->IsOp(), true,
                    platform::errors::InvalidArgument("node is not Op"));
  for (auto *n : node->outputs) {
    if (n->Name() == var_name) {
      return n;
    }
  }
  return nullptr;
}

void IpuInplacePass::ApplyImpl(ir::Graph *graph) const {
  // use this pass after forward_graph_extract_pass
  // raise error if the inplaced var both in feed_list & fetch_list
  VLOG(10) << "enter IpuInplacePass::ApplyImpl";
  VLOG(10) << "Raw Graph: ";
  VLOG(10) << DebugString(graph);

  std::vector<std::string> feed_list;
  feed_list = Get<std::vector<std::string>>("feed_list");
  std::vector<std::string> fetch_list;
  fetch_list = Get<std::vector<std::string>>("fetch_list");

  bool is_feed = false;
  bool is_fetch = false;
  ir::Node *var = nullptr;
  auto RenameInplaceVar = [&](ir::Node *node) {
    auto *op = node->Op();
    for (auto name : op->Input("__inputs__")) {
      for (auto name_out : op->Output("__outputs__")) {
        if (name == name_out) {
          is_feed = std::find(feed_list.begin(), feed_list.end(), name) !=
                    feed_list.end();
          is_fetch = std::find(fetch_list.begin(), fetch_list.end(), name) !=
                     fetch_list.end();
          auto new_name = name + "__inplace_1";
          if (!is_feed && !is_fetch) {
            VLOG(10) << "replace op node: " << node->Name()
                     << " output var: " << name << " to " << new_name;
            var = GetOutVarNode(name, node);
          } else if (is_feed) {
            VLOG(10) << "replace op node: " << node->Name()
                     << " output var: " << name << " to " << new_name;
            var = GetOutVarNode(name, node);
          } else if (is_fetch) {
            VLOG(10) << "replace op node: " << node->Name()
                     << " input var: " << name << " to " << new_name;
            var = GetInVarNode(name, node);
          } else {
            PADDLE_THROW(platform::errors::Unimplemented(
                "found inplace op: %s, i/o var: %s, i/o are both feed/fetch "
                "vars",
                op->Type(), name));
          }

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
  };

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

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(ipu_inplace_pass, paddle::framework::ir::IpuInplacePass)
    .RequirePassAttr("feed_list")
    .RequirePassAttr("fetch_list");
