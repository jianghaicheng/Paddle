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

#include "paddle/fluid/framework/ir/ipu/inference_graph_extract_pass.h"

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {

void InferenceGraphExtractPass::ApplyImpl(ir::Graph* graph) const {
  VLOG(10) << "enter InferenceGraphExtractPass::ApplyImpl";

  // // graph_viz_pass
  // auto graph_viz_pass = PassRegistry::Instance().Get("graph_viz_pass");
  // graph_viz_pass->Set("graph_viz_path",
  //                    new std::string("before_pass.dot"));
  // graph_viz_pass->Apply(graph);

  std::unique_ptr<ir::Node> feed_var;
  std::unique_ptr<ir::Node> fetch_var;
  std::map<std::string, std::unique_ptr<ir::Node>> feed_ops = {};
  std::map<std::string, std::unique_ptr<ir::Node>> fetch_ops = {};

  std::vector<std::string> feed_list = {};
  std::vector<std::string> fetch_list = {};
  std::unordered_set<ir::Node*> feed_fetch_nodes = {};

  // Get feed_list and fetch_list
  auto batch_size = int64_t(graph->Get<int>("batch_size"));
  for (auto node : graph->Nodes()) {
    if (node->Name() == "feed") {
      if (node->IsOp()) {
        feed_list.push_back(node->outputs[0]->Name());
        // Make the batch_size fixed
        auto input_shape = node->outputs[0]->Var()->GetShape();
        input_shape[0] = batch_size;
        node->outputs[0]->Var()->SetShape(input_shape);
      }
      feed_fetch_nodes.insert(node);
    } else if (node->Name() == "fetch") {
      if (node->IsOp()) {
        fetch_list.push_back(node->inputs[0]->Name());
      }
      feed_fetch_nodes.insert(node);
    }
  }

  // save feed and fetch nodes
  for (auto node : feed_fetch_nodes) {
    if (node->Name() == "feed") {
      if (node->IsOp()) {
        // int64->int32
        if (node->outputs[0]->Var()->GetDataType() == proto::VarType::INT64) {
          node->outputs[0]->Var()->SetDataType(proto::VarType::INT32);
        }
        node->outputs[0]->inputs.clear();
        feed_ops.emplace(node->outputs[0]->Name(), graph->RemoveNode(node));
      } else {
        feed_var.reset(graph->RemoveNode(node).release());
      }
    }
    if (node->Name() == "fetch") {
      if (node->IsOp()) {
        node->inputs[0]->outputs.clear();
        fetch_ops.emplace(node->inputs[0]->Name(), graph->RemoveNode(node));
      } else {
        fetch_var.reset(graph->RemoveNode(node).release());
      }
    }
  }

  auto inference_compile_pass =
      PassRegistry::Instance().Get("inference_compile_pass");
  inference_compile_pass->Set(
      "feed_list",
      new std::vector<std::string>(feed_list.begin(), feed_list.end()));
  inference_compile_pass->Set(
      "fetch_list",
      new std::vector<std::string>(fetch_list.begin(), fetch_list.end()));
  inference_compile_pass->Apply(graph);

  graph->AddNode(feed_var.release());
  graph->AddNode(fetch_var.release());
  for (auto feed_name : feed_list) {
    for (auto node : graph->Nodes()) {
      if (node->Name() == feed_name) {
        auto feed_op_node = graph->AddNode(feed_ops.at(feed_name).release());
        node->inputs.push_back(feed_op_node);
      }
    }
  }
  for (auto fetch_name : fetch_list) {
    for (auto node : graph->Nodes()) {
      if (node->Name() == fetch_name) {
        auto fetch_op_node = graph->AddNode(fetch_ops.at(fetch_name).release());
        node->outputs.push_back(fetch_op_node);
      }
    }
  }

  // // graph_viz_pass
  // graph_viz_pass->Erase("graph_viz_path");
  // graph_viz_pass->Set("graph_viz_path",
  //                     new std::string("after_pass.dot"));
  // graph_viz_pass->Apply(graph);

  VLOG(10) << "leave InferenceGraphExtractPass::ApplyImpl";
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(inference_graph_extract_pass,
              paddle::framework::ir::InferenceGraphExtractPass);
USE_PASS(graph_viz_pass);