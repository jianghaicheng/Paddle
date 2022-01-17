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

/**
 * This File should be automatically generated by coding auto generator.
 * All ops C++ autograd logic is defined here, in Python-C extension API
 * system we try to avoid any autograd related code, and move them all to
 * here.
 *
 * Currently, we just manually do some fwd autograd here. And we will replace
 * them with auto code generator later.
 * **/

#include "paddle/fluid/eager/api/generated/eager_generated/forwards/scale.h"
#include "paddle/fluid/eager/api/generated/eager_generated/backwards/scale_node.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/eager/utils.h"

#include "paddle/pten/api/all.h"

namespace egr {

egr::EagerTensor scale(const egr::EagerTensor& x, float scale, float bias,
                       bool bias_after_scale, bool trace_backward) {
  // 1. Run Forward
  // 1.1 Create outputs
  egr::EagerTensor out;
  // 1.2 Need by original op, we assemble ins, outs, attrs here

  // 1.3 Call forward C++ api
  ScaleAPI(x, scale, bias, bias_after_scale, &out);

  // 2. Build Backward Depends
  // 2.1 Get AutogradMetas for all ins and outs
  auto p_autograd_in = EagerUtils::unsafe_autograd_meta(x);
  // NOTE: Call EagerUtils::multi_autograd_meta  when we have vector of outputs
  auto p_autograd_out = EagerUtils::autograd_meta(&out);

  // 2.2 Add GradNode
  // 2.2.1 ComputeRequireGrad
  // TODO(jiabin) : make this function accept different kinds of input
  // TODO(zhanlve): which one is more efficient:
  //                1. construct a vector of pointers
  //                2. call "ComputeRequireGrad" multiple times
  bool require_any_grad =
      EagerUtils::ComputeRequireGrad(trace_backward, p_autograd_in);
  if (require_any_grad) {
    EagerUtils::PassStopGradient(false /*generate_grad*/, p_autograd_out);

    // 2.2.2 Set OutRankInfo for outputs this needs to be as same as Edges's
    // input_rank_
    /** Note:
    // 1. We provide EagerUtils::SetMultiOutRank(vector<AutogradMeta*>),
    // since we have some of Operator has servel slot name with duplicate
    outputs.
    // 2. We call AutogradMeta's SetOutput Rank only when we have single output
    with
    // single slot name.
    **/
    p_autograd_out->SetSingleOutRankWithSlot(0, 0);

    // Init GradNode
    auto scale_node = std::make_shared<GradNodeScale>(/* fwd_in_slot_num */ 1,
                                                      /* bwd_in_slot_num */ 1);

    // Pass Attributes to GradNode
    scale_node->SetAttributes_scale(scale);

    // Set Next Edges
    scale_node->AddEdges(p_autograd_in, /*slot id*/ 0);

    // Set TensorWrappers
    scale_node->SetTensorWrappers_X({x});

    // Set Grad out rank as same as fwd input and set stop gradient to bwd
    scale_node->SetGradOutMeta(*p_autograd_in, /*slot id*/ 0);
    // Set Grad out rank as same as fwd input and set stop gradient to bwd
    scale_node->SetGradInMeta(*p_autograd_out, /*slot id*/ 0);

    // Set History for output set current Grad Node for
    EagerUtils::SetHistory(p_autograd_out, scale_node);
  }

  return out;
}

}  // namespace egr
