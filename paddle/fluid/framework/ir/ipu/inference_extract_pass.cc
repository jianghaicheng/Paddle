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

#include "paddle/fluid/framework/ir/ipu/inference_extract_pass.h"

#include "paddle/fluid/framework/ipu/ipu_backend.h"
#include "paddle/fluid/framework/ipu/ipu_strategy.h"

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {

void InferenceExtractPass::ApplyImpl(ir::Graph* graph) const {
  VLOG(10) << "enter InferenceExtractPass::ApplyImpl";
  VLOG(10) << "Raw Graph: ";
  VLOG(10) << DebugString(graph);

  // // graph_viz_pass
  // auto graph_viz_pass = PassRegistry::Instance().Get("graph_viz_pass");
  // graph_viz_pass->Set("graph_viz_path",
  //                     new std::string("before_pass.dot"));
  // graph_viz_pass->Apply(graph);

  std::shared_ptr<ipu::IpuBackend> ipu_backend = ipu::IpuBackend::GetInstance();

  // Get scope
  if (graph->Has(kParamScopeAttr)) {
    auto& scope = graph->Get<Scope>(kParamScopeAttr);
    ipu_backend->SetScope(scope);
  } else {
    PADDLE_THROW(platform::errors::Unimplemented("Can not find the scope."));
  }

  // Get feed_list and fetch_list
  std::vector<std::string> feed_list = {};
  std::vector<std::string> fetch_list = {};
  for (auto node : graph->Nodes()) {
    if (node->IsOp()) {
      if (node->Name() == "feed") {
        feed_list.push_back(node->outputs[0]->Name());
      }
      if (node->Name() == "fetch") {
        fetch_list.push_back(node->inputs[0]->Name());
      }
    }
  }

  // TODO(yaozhixin): ipu_backend manages ipu_strategy
  static std::shared_ptr<ipu::IpuStrategy> ipu_strategy_instance_(
      new ipu::IpuStrategy());

  ipu_strategy_instance_->is_training = false;
  ipu_strategy_instance_->num_ipus = graph->Get<int>("num_ipus");
  ipu_strategy_instance_->popart_options_.enablePipelining =
      graph->Get<bool>("enable_pipeline");
  if (ipu_strategy_instance_->num_ipus > 1) {
    ipu_strategy_instance_->popart_options_.virtualGraphMode =
        ipu::VirtualGraphMode::Auto;
  } else {
    ipu_strategy_instance_->popart_options_.virtualGraphMode =
        ipu::VirtualGraphMode::Off;
  }

  ipu_backend->SetIpuStrategy(*(ipu_strategy_instance_.get()));

  // Remove useless vars
  std::unordered_set<const Node*> useless_nodes;
  for (auto node : graph->Nodes()) {
    if (!node->inputs.size() && !node->outputs.size()) {
      useless_nodes.insert(node);
    }
  }
  GraphSafeRemoveNodes(graph, useless_nodes);

  auto ipu_graph_builder_pass =
      PassRegistry::Instance().Get("ipu_graph_builder_pass");
  ipu_graph_builder_pass->Set(
      "feed_list",
      new std::vector<std::string>(feed_list.begin(), feed_list.end()));
  ipu_graph_builder_pass->Set(
      "fetch_list",
      new std::vector<std::string>(fetch_list.begin(), fetch_list.end()));
  ipu_graph_builder_pass->Apply(graph);

  auto ipu_runtime_replacer_pass =
      PassRegistry::Instance().Get("ipu_runtime_replacer_pass");
  ipu_runtime_replacer_pass->Set(
      "feed_list",
      new std::vector<std::string>(feed_list.begin(), feed_list.end()));
  ipu_runtime_replacer_pass->Set(
      "fetch_list",
      new std::vector<std::string>(fetch_list.begin(), fetch_list.end()));
  ipu_runtime_replacer_pass->Apply(graph);

  // // graph_viz_pass
  // graph_viz_pass->Erase("graph_viz_path");
  // graph_viz_pass->Set("graph_viz_path",
  //                     new std::string("after_pass.dot"));
  // graph_viz_pass->Apply(graph);

  VLOG(10) << "Post Graph: ";
  VLOG(10) << DebugString(graph);
  VLOG(10) << "leave InferenceExtractPass::ApplyImpl";
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(inference_extract_pass,
              paddle::framework::ir::InferenceExtractPass);
USE_PASS(graph_viz_pass);