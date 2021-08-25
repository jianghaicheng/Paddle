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

#include <string>

#include "paddle/fluid/framework/ipu/ipu_backend.h"
#include "paddle/fluid/framework/ipu/ipu_strategy.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/enforce.h"

// debug
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {

void InferenceExtractPass::ApplyImpl(ir::Graph* graph) const {
  VLOG(10) << "enter InferenceExtractPass::ApplyImpl";
  VLOG(10) << "Raw Graph: ";
  VLOG(10) << DebugString(graph);

  std::shared_ptr<ipu::IpuBackend> ipu_backend = ipu::IpuBackend::GetInstance();

  // Get scope
  if (graph->Has(kParamScopeAttr)) {
    auto& scope = graph->Get<Scope>(kParamScopeAttr);
    ipu_backend->SetScope(scope);
  } else {
    PADDLE_THROW(platform::errors::Unimplemented("Can not find the scope."));
  }

  // TODO(yaozhixin): ipu_backend manages ipu_strategy
  static std::shared_ptr<ipu::IpuStrategy> ipu_strategy_instance_(
      new ipu::IpuStrategy());

  ipu_strategy_instance_->is_training = false;
  ipu_strategy_instance_->num_ipus = graph->Get<int>("num_ipus");
  ipu_strategy_instance_->popart_options_.enablePipelining =
      graph->Get<bool>("enable_pipeline");
  auto& enable_sharding = graph->Get<bool>("enable_sharding");
  if (enable_sharding) {
    ipu_strategy_instance_->popart_options_.virtualGraphMode =
        ipu::VirtualGraphMode::Manual;
  } else {
    ipu_strategy_instance_->popart_options_.virtualGraphMode =
        ipu::VirtualGraphMode::Off;
  }

  ipu_backend->SetIpuStrategy(*(ipu_strategy_instance_.get()));

  VLOG(10) << "Post Graph: ";
  VLOG(10) << DebugString(graph);
  VLOG(10) << "leave InferenceExtractPass::ApplyImpl";
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(inference_extract_pass,
              paddle::framework::ir::InferenceExtractPass);