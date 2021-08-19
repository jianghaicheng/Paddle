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

#pragma once

#include "paddle/fluid/framework/ipu/popart_canonicalization/canonicalization_utils.h"

namespace paddle {
namespace framework {
namespace ipu {

std::string GenerateVarName();

ir::Node *MakeVarNode(ir::Graph *graph);
ir::Node *MakeOpNode(ir::Graph *graph, const std::string &type,
                     const std::vector<ir::Node *> &inputs,
                     const std::vector<ir::Node *> &outputs);

Node *CreateBaseOp(ir::Graph *graph, const std::string &type,
                   const std::vector<ir::Node *> &inputs,
                   const std::vector<ir::Node *> &outputs,
                   const AttributeMap &attrs = {});

template <typename T>
AttributeMap MakeConstAttrMap(std::vector<T> value, std::vector<int64_t> dims,
                              int dtype) {
  return AttributeMap{{"value", value}, {"dims", dims}, {"dtype", dtype}};
}

template <typename T>
AttributeMap MakeConstAttrMapFromValue(T v, std::vector<int64_t> dims,
                                       int dtype) {
  size_t size = 1;
  for (auto &dim : dims) {
    size *= dim;
  }
  return MakeConstAttrMap<T>(std::vector<T>(size, v), dims, dtype);
}

ir::Node *CreateConst(ir::Graph *graph, const std::vector<ir::Node *> &inputs,
                      const std::vector<ir::Node *> &outputs,
                      const AttributeMap &attrs);

// otype is proto::VarType::Type
ir::Node *CreateCast(ir::Graph *graph, const std::vector<ir::Node *> &inputs,
                     const std::vector<ir::Node *> &outputs, const int &otype);

}  // namespace ipu
}  // namespace framework
}  // namespace paddle
