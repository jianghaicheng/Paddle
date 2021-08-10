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

namespace paddle {
namespace framework {

// onnx dtype
// https://github.com/onnx/onnx/blob/master/onnx/onnx-ml.proto3
enum ONNXDataType : int {
  UNDEFINED = 0,
  FLOAT = 1,
  UINT8 = 2,
  INT8 = 3,
  UINT16 = 4,
  INT16 = 5,
  INT32 = 6,
  INT64 = 7,
  STRING = 8,
  BOOL = 9,
  FLOAT16 = 10,
  DOUBLE = 11,
  UINT32 = 12,
  UINT64 = 13,
  COMPLEX64 = 14,
  COMPLEX128 = 15,
  BFLOAT16 = 16
};

// This avoids the static initialisation order fiasco,
std::unordered_map<std::string, SymbolHandler> &SymbolHandlers() {
  static std::unordered_map<std::string, SymbolHandler> symbol_handlers;
  return symbol_handlers;
}

bool RegisterHandler(const std::string &symbol, const SymbolHandler &handler) {
  if (SymbolHandlers().count(symbol) != 0) {
    LOG(WARNING) << "Trying to register popart handler twice for operator: "
                 << symbol;
    return false;
  }
  bool new_handler = SymbolHandlers().emplace(symbol, handler).second;
  return new_handler;
}

// Return a pointer to a handler if one is registered for this kind of node or
// an empty std::function otherwise.
SymbolHandler GetHandler(const std::string &kind) {
  auto it = SymbolHandlers().find(kind);
  if (it != SymbolHandlers().end()) {
    return it->second;
  }
  return {};
}

void MoveNodeInputs(ir::Node *node, ir::Node *new_node) {
  if (node->inputs.empty()) {
    return;
  }
  new_node->inputs = node->inputs;
  for (auto *node_in : node->inputs) {
    for (size_t i = 0; i < node_in->outputs.size(); ++i) {
      if (node_in->outputs[i] == node) {
        node_in->outputs[i] = new_node;
        break;
      }
    }
  }
}

void MoveNodeOutputs(ir::Node *node, ir::Node *new_node) {
  if (node->outputs.empty()) {
    return;
  }
  new_node->outputs = node->outputs;
  for (auto *node_out : node->outputs) {
    for (size_t i = 0; i < node_out->inputs.size(); ++i) {
      if (node_out->inputs[i] == node) {
        node_out->inputs[i] = new_node;
        break;
      }
    }
  }
}

void ConnectNodes(ir::Node *first_node, ir::Node *next_node) {
  first_node->outputs.push_back(next_node);
  next_node->inputs.push_back(first_node);
}

int ConvertDataType(int type) {
  auto dtype = static_cast<proto::VarType::Type>(type);
  switch (dtype) {
    case proto::VarType::BOOL:
      return static_cast<int>(ONNXDataType::BOOL);
    case proto::VarType::INT16:
      return static_cast<int>(ONNXDataType::INT16);
    case proto::VarType::INT32:
      return static_cast<int>(ONNXDataType::INT32);
    case proto::VarType::INT64:
      return static_cast<int>(ONNXDataType::INT64);
    case proto::VarType::FP16:
      return static_cast<int>(ONNXDataType::FLOAT16);
    case proto::VarType::FP32:
      return static_cast<int>(ONNXDataType::FLOAT);
    case proto::VarType::FP64:
      return static_cast<int>(ONNXDataType::DOUBLE);
    case proto::VarType::UINT8:
      return static_cast<int>(ONNXDataType::UINT8);
    case proto::VarType::INT8:
      return static_cast<int>(ONNXDataType::INT8);
    case proto::VarType::BF16:
      return static_cast<int>(ONNXDataType::BFLOAT16);
    case proto::VarType::COMPLEX64:
      return static_cast<int>(ONNXDataType::COMPLEX64);
    case proto::VarType::COMPLEX128:
      return static_cast<int>(ONNXDataType::COMPLEX128);
    default:
      PADDLE_THROW(
          platform::errors::Unimplemented("Unsupported data type: %d.", dtype));
  }
}

}  // namespace framework
}  // namespace paddle
