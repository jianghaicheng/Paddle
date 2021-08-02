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

#include "paddle/fluid/framework/ir/ipu/forward_graph_extract_pass.h"

#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <iosfwd>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {

void ForwardGraphExtractPass::ApplyImpl(ir::Graph* graph) const {
  VLOG(10) << "enter ForwardGraphExtractPass::ApplyImpl";
  // find forward ops
  // find forward vars
  // remove unneeded nodes
  // topology_sort forward_ops(outside this Pass)
  // remove unneeded vars in scope outside this Pass

  VLOG(10) << "Raw Graph: ";
  VLOG(10) << DebugString(graph);

  // // graph_viz_pass
  // auto graph_viz_pass = PassRegistry::Instance().Get("graph_viz_pass");
  // graph_viz_pass->Set("graph_viz_path",
  //                     new std::string("/home/Paddle/demos/before_pass.dot"));
  // graph_viz_pass->Apply(graph);

  std::unordered_map<OpRole, std::unordered_set<ir::Node*>> all_ops{
      {OpRole::kForward, {}},  {OpRole::kBackward, {}},
      {OpRole::kOptimize, {}}, {OpRole::kRPC, {}},
      {OpRole::kDist, {}},     {OpRole::kLRSched, {}},
      {OpRole::kLoss, {}},     {OpRole::kNotSpecified, {}}};
  std::unordered_map<OpRole, std::unordered_set<std::string>> all_ops_name{
      {OpRole::kForward, {}},  {OpRole::kBackward, {}},
      {OpRole::kOptimize, {}}, {OpRole::kRPC, {}},
      {OpRole::kDist, {}},     {OpRole::kLRSched, {}},
      {OpRole::kLoss, {}},     {OpRole::kNotSpecified, {}}};

  for (auto* node : graph->Nodes()) {
    if (!node->IsOp()) {
      continue;
    }
    auto op_role = BOOST_GET_MUTABLE(int, node->Op()->GetAttr("op_role"));
    if (op_role == static_cast<int>(OpRole::kForward)) {
      all_ops[OpRole::kForward].insert(node);
      all_ops_name[OpRole::kForward].insert(node->Name());
    } else if (op_role == static_cast<int>(OpRole::kBackward)) {
    } else if (op_role == static_cast<int>(OpRole::kOptimize)) {
      all_ops[OpRole::kOptimize].insert(node);
      all_ops_name[OpRole::kOptimize].insert(node->Name());
    } else if (op_role == static_cast<int>(OpRole::kRPC)) {
    } else if (op_role == static_cast<int>(OpRole::kDist)) {
    } else if (op_role == static_cast<int>(OpRole::kLRSched)) {
    } else if (op_role == static_cast<int>(OpRole::kLoss)) {
      all_ops[OpRole::kLoss].insert(node);
      all_ops_name[OpRole::kLoss].insert(node->Name());
    } else if (op_role == static_cast<int>(OpRole::kNotSpecified)) {
      LOG(WARNING) << "Op: " << node->Name() << " OpRole is NotSpecified ";
    }
  }

  std::unordered_set<std::string> forward_var_names;
  std::unordered_set<ir::Node*> backward_vars;
  // std::unordered_set<ir::Node*> forward_vars;

  for (auto& nodes : std::array<std::unordered_set<ir::Node*>, 2>{
           all_ops[OpRole::kForward], all_ops[OpRole::kLoss],
           //  all_ops[OpRole::kOptimize],
       }) {
    for (auto* node : nodes) {
      for (auto& name_map : node->Op()->Inputs()) {
        for (auto& name : name_map.second) {
          forward_var_names.insert(name);
        }
      }
      for (auto& name_map : node->Op()->Outputs()) {
        for (auto& name : name_map.second) {
          forward_var_names.insert(name);
        }
      }
    }
  }

  auto not_contains = [&](const std::string& name,
                          std::unordered_set<std::string>& names) {
    return names.find(name) == names.end();
  };

  for (auto* node : graph->Nodes()) {
    if (!node->IsVar()) {
      continue;
    }
    for (auto* in_node : node->inputs) {
      if (!not_contains(in_node->Name(), all_ops_name[OpRole::kOptimize])) {
        backward_vars.insert(node);
      }
    }
  }

  std::unordered_set<ir::Node*> rm_nodes;
  for (auto* node : graph->Nodes()) {
    if (backward_vars.find(node) != backward_vars.end()) {
      rm_nodes.insert(node);
    } else if (not_contains(node->Name(), all_ops_name[OpRole::kForward]) &&
               not_contains(node->Name(), all_ops_name[OpRole::kLoss]) &&
               // not_contains(node->Name(), all_ops_name[OpRole::kOptimize]) &&
               not_contains(node->Name(), forward_var_names)) {
      rm_nodes.insert(node);
    }
  }

  VLOG(10) << "Remove Node: ";
  for (auto* node : rm_nodes) {
    // rm node releations
    for (auto* node_in : node->inputs) {
      for (size_t i = 0; i < node_in->outputs.size(); ++i) {
        if (node_in->outputs[i] == node) {
          node_in->outputs.erase(node_in->outputs.begin() + i);
          break;
        }
      }
    }
    for (auto* node_out : node->outputs) {
      for (size_t i = 0; i < node_out->inputs.size(); ++i) {
        if (node_out->inputs[i] == node) {
          node_out->inputs.erase(node_out->inputs.begin() + i);
          break;
        }
      }
    }

    VLOG(10) << "\t" << node->Name();
    graph->RemoveNode(node);
  }

  VLOG(10) << "Post Graph: ";
  VLOG(10) << DebugString(graph);

  // // graph_viz_pass
  // graph_viz_pass->Erase("graph_viz_path");
  // graph_viz_pass->Set("graph_viz_path",
  //                     new std::string("/home/Paddle/demos/after_pass.dot"));
  // graph_viz_pass->Apply(graph);

  VLOG(10) << "leave ForwardGraphExtractPass::ApplyImpl";
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(forward_graph_extract_pass,
              paddle::framework::ir::ForwardGraphExtractPass);

USE_PASS(graph_viz_pass);
