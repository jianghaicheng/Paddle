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

#include "paddle/fluid/framework/ir/ipu/optimizer_extract_pass.h"

#include "paddle/fluid/framework/ipu/ipu_backend.h"
#include "paddle/fluid/framework/op_proto_maker.h"

namespace paddle {
namespace framework {
namespace ir {

class Graph;

void IpuOptimizerExtractPass::ApplyImpl(ir::Graph* graph) const {
  auto ipu_backend = paddle::framework::IpuBackend::GetInstance();

  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op()) {
      int op_role = BOOST_GET_CONST(
          int, node->Op()->GetAttr(
                   framework::OpProtoAndCheckerMaker::OpRoleAttrName()));

      // graph usually have multiple optimizer node for different parameter,
      // and these node have the same type and attr value usually
      if ((op_role == static_cast<int>(framework::OpRole::kOptimize))) {
        ipu_backend->SetOptimizerType(node->Op()->Type());

        for (const std::string& attr_name : node->Op()->AttrNames()) {
          auto attr_type = node->Op()->GetAttrType(attr_name);
          // with adam, attr are float
          if (attr_type == proto::AttrType::FLOAT) {
            auto attr_value =
                BOOST_GET_CONST(float, node->Op()->GetAttr(attr_name));
            ipu_backend->SetOptimizerAttr(attr_name, attr_value);
          } else {
            VLOG(10) << "Skip " << attr_type;
          }
        }
      }
    }
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(optimizer_extract_pass,
              paddle::framework::ir::IpuOptimizerExtractPass);