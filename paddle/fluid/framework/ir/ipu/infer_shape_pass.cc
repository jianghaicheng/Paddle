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

#include "paddle/fluid/framework/ir/ipu/infer_shape_pass.h"

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/ipu/ipu_backend.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/variable_helper.h"

namespace paddle {
namespace framework {
namespace ir {

void InferShapePass::ApplyImpl(ir::Graph* graph) const {
  VLOG(10) << "enter InferShapePass::ApplyImpl";

  // graph_viz_pass
  // auto graph_viz_pass = PassRegistry::Instance().Get("graph_viz_pass");
  // graph_viz_pass->Set("graph_viz_path",
  //                    new std::string("before_pass.dot"));
  // graph_viz_pass->Apply(graph);

  std::shared_ptr<ipu::IpuBackend> ipu_backend = ipu::IpuBackend::GetInstance();

  if (ipu_backend->GetIpuStrategy()->need_infer_shape) {
    VLOG(10) << "start InferShapePass";
    // temp scope for shape inference
    std::shared_ptr<paddle::framework::Scope> scope(
        new paddle::framework::Scope());
    for (auto node : graph->Nodes()) {
      if (!node->IsVar()) {
        continue;
      }
      auto var_desc = node->Var();
      auto* ptr = scope->Var(var_desc->Name());
      paddle::framework::InitializeVariable(ptr, var_desc->GetType());

      auto tensor = ptr->GetMutable<paddle::framework::LoDTensor>();
      tensor->Resize(paddle::framework::make_ddim(var_desc->GetShape()));

      // skip lowerWeights in IpuBackend if persistable node has inputs
      if (var_desc->Persistable() && node->inputs.size()) {
        var_desc->SetPersistable(false);
      }
    }

    // infer shape
    auto nodes = ir::TopologySortOperations(*graph);
    for (auto node : nodes) {
      auto op_desc = node->Op();
      auto op = paddle::framework::OpRegistry::CreateOp(*op_desc);
      paddle::framework::RuntimeContext ctx(op->Inputs(), op->Outputs(),
                                            *scope);
      op->RuntimeInferShape(*scope, paddle::platform::CPUPlace(), ctx);

      for (auto it = ctx.outputs.begin(); it != ctx.outputs.end(); it++) {
        for (int i = 0; i < it->second.size(); i++) {
          auto output_name = op_desc->Output(it->first)[i];
          auto dim =
              it->second[i]->GetMutable<paddle::framework::LoDTensor>()->dims();
          auto new_shape = paddle::framework::vectorize(dim);
          for (auto output_node : node->outputs) {
            if (output_node->Name() == output_name) {
              output_node->Var()->SetShape(new_shape);
            }
          }
        }
      }
    }
    // release the temp scope
    scope.reset();
    VLOG(10) << "end InferShapePass";
  }

  // graph_viz_pass
  // graph_viz_pass->Erase("graph_viz_path");
  // graph_viz_pass->Set("graph_viz_path",
  //                     new std::string("after_pass.dot"));
  // graph_viz_pass->Apply(graph);

  VLOG(10) << "leave InferShapePass::ApplyImpl";
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(infer_shape_pass, paddle::framework::ir::InferShapePass);