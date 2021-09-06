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

#include "paddle/fluid/framework/ir/ipu/inference_attr_extract_pass.h"

#include "paddle/fluid/framework/ipu/ipu_backend.h"
#include "paddle/fluid/framework/ipu/ipu_strategy.h"

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

void InferenceAttrExtractPass::ApplyImpl(ir::Graph* graph) const {
  VLOG(10) << "enter InferenceAttrExtractPass::ApplyImpl";

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
  auto num_ipus = graph->Get<int>("num_ipus");
  ipu_strategy_instance_->num_ipus = num_ipus;
  if (num_ipus > 1) {
    ipu_strategy_instance_->popart_options_.virtualGraphMode =
        ipu::VirtualGraphMode::Manual;
  } else {
    ipu_strategy_instance_->popart_options_.virtualGraphMode =
        ipu::VirtualGraphMode::Off;
  }

  auto enable_pipelining = graph->Get<bool>("enable_pipelining");
  ipu_strategy_instance_->popart_options_.enablePipelining = enable_pipelining;
  if (enable_pipelining) {
    auto batches_per_step = graph->Get<int>("batches_per_step");
    PADDLE_ENFORCE_GE(
        batches_per_step, num_ipus,
        platform::errors::InvalidArgument("Batched per step should be equal or "
                                          "greater than the number of IPUs"));
    ipu_strategy_instance_->batches_per_step = batches_per_step;
  }

  ipu_backend->SetIpuStrategy(*(ipu_strategy_instance_.get()));

  VLOG(10) << "leave InferenceAttrExtractPass::ApplyImpl";
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(inference_attr_extract_pass,
              paddle::framework::ir::InferenceAttrExtractPass);