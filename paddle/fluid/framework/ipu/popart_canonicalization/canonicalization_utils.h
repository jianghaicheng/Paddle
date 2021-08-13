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

#include "paddle/fluid/framework/ipu/ipu_utils.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ipu {

// TODO(alleng) remove ir::
using ir::Graph;
using ir::Node;

#define REGISTER_HANDLER(name, func) \
  static bool __UNUSED_##name =      \
      paddle::framework::ipu::RegisterHandler(#name, func)

using SymbolHandler = std::function<ir::Node *(ir::Graph *, ir::Node *)>;

std::unordered_map<std::string, SymbolHandler> &SymbolHandlers();

bool RegisterHandler(const std::string &, const SymbolHandler &);

SymbolHandler GetHandler(const std::string &);

// TODO(alleng) remove these func
void MoveNodeInputs(ir::Node *node, ir::Node *new_node);
void MoveNodeOutputs(ir::Node *node, ir::Node *new_node);

void ReplaceNodeInputs(ir::Node *node, ir::Node *new_node);
void ReplaceNodeOutputs(ir::Node *node, ir::Node *new_node);
void ConnectNodes(ir::Node *first_node, ir::Node *next_node);
void CopyOpAttr(const std::string &attr_name, OpDesc *op, OpDesc *new_op,
                bool override = false);

const int ConvertDataType(const int &type);

Node *GetInputNode(const std::string &name, const Node *node);

}  // namespace ipu
}  // namespace framework
}  // namespace paddle
