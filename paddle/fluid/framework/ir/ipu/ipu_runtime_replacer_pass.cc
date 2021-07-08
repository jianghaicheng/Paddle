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

#include <algorithm>
#include <array>
#include <fstream>
#include <iosfwd>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/ipu/ipu_pass_base.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

// debug
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {

class Graph;

class IpuRuntimeReplacerPass : public IPUPassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;
};

void IpuRuntimeReplacerPass::ApplyImpl(ir::Graph* graph) const {
  VLOG(10) << "enter IpuRuntimeReplacerPass::ApplyImpl";

  VLOG(10) << "Raw Graph: ";
  VLOG(10) << DebugString(graph);

  // graph_viz_pass
  auto graph_viz_pass = PassRegistry::Instance().Get("graph_viz_pass");
  graph_viz_pass->Set("graph_viz_path",
                      new std::string("/home/Paddle/demos/before_ipu_runtime_replacer_pass.dot"));
  graph_viz_pass->Apply(graph);

  std::vector<std::string> feed_list;
  feed_list = Get<std::vector<std::string>>("feed_list");

  std::vector<std::string> fetch_list;
  fetch_list = Get<std::vector<std::string>>("fetch_list");

  framework::OpDesc new_op_desc;
  new_op_desc.SetType("ipu_runtime");
  new_op_desc.SetInput("FeedList", feed_list);
  new_op_desc.SetOutput("FetchList", fetch_list);
  new_op_desc.Flush();

  // Create a new node for the ipu_runtime_op.
  graph->CreateOpNode(&new_op_desc);

  // Remove unneeded nodes.
  std::unordered_set<const Node*> marked_nodes;
  for (auto* node : graph->Nodes()) {
    if (node->IsOp()) {
      auto* op_desc = node->Op();
      if (op_desc->Type() != "ipu_runtime") {
        marked_nodes.insert(node);
      }
    }
  }

  GraphSafeRemoveNodes(graph, marked_nodes);

  VLOG(10) << "Post Graph: ";
  VLOG(10) << DebugString(graph);

  // graph_viz_pass
  graph_viz_pass->Erase("graph_viz_path");
  graph_viz_pass->Set("graph_viz_path",
                      new std::string("/home/Paddle/demos/after_ipu_runtime_replacer_pass.dot"));
  graph_viz_pass->Apply(graph);

  VLOG(10) << "leave IpuRuntimeReplacerPass::ApplyImpl";
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(ipu_runtime_replacer_pass,
              paddle::framework::ir::IpuRuntimeReplacerPass)
    .RequirePassAttr("feed_list")
    .RequirePassAttr("fetch_list");

USE_PASS(graph_viz_pass);
