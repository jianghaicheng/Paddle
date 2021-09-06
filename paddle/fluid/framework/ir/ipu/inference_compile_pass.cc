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

#include "paddle/fluid/framework/ir/ipu/inference_compile_pass.h"

#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {

void InferenceCompilePass::ApplyImpl(ir::Graph* graph) const {
  VLOG(10) << "enter InferenceCompilePass::ApplyImpl";
  VLOG(10) << "Raw Graph: ";
  VLOG(10) << DebugString(graph);

  // // graph_viz_pass
  // auto graph_viz_pass = PassRegistry::Instance().Get("graph_viz_pass");
  // graph_viz_pass->Set("graph_viz_path",
  //                    new std::string("/before_pass.dot"));
  // graph_viz_pass->Apply(graph);

  std::vector<std::string> feed_list;
  feed_list = Get<std::vector<std::string>>("feed_list");
  std::vector<std::string> fetch_list;
  fetch_list = Get<std::vector<std::string>>("fetch_list");

  auto forward_graph_extract_pass =
      PassRegistry::Instance().Get("forward_graph_extract_pass");
  forward_graph_extract_pass->Apply(graph);

  auto popart_canonicalization_pass =
      PassRegistry::Instance().Get("popart_canonicalization_pass");
  popart_canonicalization_pass->Apply(graph);

  std::vector<std::string> compile_pass = {"ipu_inplace_pass",
                                           "ipu_graph_builder_pass",
                                           "ipu_runtime_replacer_pass"};
  for (auto pass_name : compile_pass) {
    auto pass = PassRegistry::Instance().Get(pass_name);
    pass->Set("feed_list",
              new std::vector<std::string>(feed_list.begin(), feed_list.end()));
    pass->Set("fetch_list", new std::vector<std::string>(fetch_list.begin(),
                                                         fetch_list.end()));
    pass->Apply(graph);
  }

  // // graph_viz_pass
  // graph_viz_pass->Erase("graph_viz_path");
  // graph_viz_pass->Set("graph_viz_path",
  //                     new std::string("after_pass.dot"));
  // graph_viz_pass->Apply(graph);

  VLOG(10) << "Post Graph: ";
  VLOG(10) << DebugString(graph);
  VLOG(10) << "leave InferenceCompilePass::ApplyImpl";
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(inference_compile_pass,
              paddle::framework::ir::InferenceCompilePass)
    .RequirePassAttr("feed_list")
    .RequirePassAttr("fetch_list");
USE_PASS(graph_viz_pass);