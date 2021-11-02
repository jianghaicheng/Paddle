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

#include "paddle/fluid/framework/ipu/popart_canonicalization/canonicalization_utils.h"
#include "paddle/fluid/framework/ipu/popart_canonicalization/op_builder.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ipu {
namespace {

Node *custom_op_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto attrs = op->GetAttrMap();
  attrs.insert({"__op_type", node->Op()->Type()});
  auto new_node = CreateBaseOp(graph, node, "popart_custom_op", node->inputs,
                               node->outputs, attrs);
  return new_node;
}

REGISTER_HANDLER(custom_op, custom_op_handler);

}  // namespace
}  // namespace ipu
}  // namespace framework
}  // namespace paddle