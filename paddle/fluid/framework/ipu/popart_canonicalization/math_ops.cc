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

ir::Node *reduce_mean_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();
  auto attrs = AttributeMap{};
  auto reduce_all = BOOST_GET_CONST(bool, op->GetAttr("reduce_all"));
  if (!reduce_all) {
    auto axes_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("dim"));
    auto axes = std::vector<int64_t>{axes_.begin(), axes_.end()};
    attrs.emplace("axes", axes);
  }
  auto keepdims_ = BOOST_GET_CONST(bool, op->GetAttr("keep_dim"));
  auto keepdims = int64_t{keepdims_};
  attrs.emplace("keepdims", keepdims);
  return CreateBaseOp(graph, "popart_reducemean", node->inputs, node->outputs,
                      attrs);
}

ir::Node *mean_handler(ir::Graph *graph, ir::Node *node) {
  return CreateBaseOp(graph, "popart_reducemean", {GetInputNode("X", node)},
                      {GetOutputNode("Out", node)},
                      {
                          {"keepdims", int64_t{0}},
                      });
}

ir::Node *pow_handler(ir::Graph *graph, ir::Node *node) {
  // Op(pow) -> Op(Constant)->Var(const_out)->Op(Pow)
  auto *op = node->Op();
  auto value_ = BOOST_GET_CONST(float, op->GetAttr("factor"));
  auto attrs =
      MakeConstAttrMapFromValue<float>(value_, {1}, ONNXDataType::FLOAT);
  auto new_node_const = CreateConst(graph, {}, {}, attrs);
  return CreateBaseOp(graph, "popart_pow",
                      {GetInputNode("X", node), new_node_const->outputs[0]},
                      node->outputs);
}

ir::Node *mul_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();
  auto x_num_col_dims = BOOST_GET_CONST(int, op->GetAttr("x_num_col_dims"));
  auto y_num_col_dims = BOOST_GET_CONST(int, op->GetAttr("y_num_col_dims"));
  if (x_num_col_dims != 1 || y_num_col_dims != 1) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "mul with x_num_col_dims or y_num_col_dims != 1"));
  }
  return CreateBaseOp(graph, "popart_matmul",
                      {GetInputNode("X", node), GetInputNode("Y", node)},
                      node->outputs);
}

ir::Node *matmul_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();
  auto transpose_x = BOOST_GET_CONST(bool, op->GetAttr("transpose_X"));
  auto transpose_y = BOOST_GET_CONST(bool, op->GetAttr("transpose_Y"));
  auto alpha = BOOST_GET_CONST(float, op->GetAttr("alpha"));
  return CreateGemm(graph, node->inputs, node->outputs, transpose_x,
                    transpose_y, alpha);
}

ir::Node *sum_handler(ir::Graph *graph, ir::Node *node) {
  return CreateBaseOp(graph, "popart_sum", node->inputs, node->outputs);
}

ir::Node *softmax_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();
  auto axis_ = BOOST_GET_CONST(int, op->GetAttr("axis"));
  auto axis = int64_t{axis_};
  return CreateBaseOp(graph, "popart_softmax", node->inputs, node->outputs,
                      {
                          {"axis", axis},
                      });
}

ir::Node *scale_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();
  auto scale_ = BOOST_GET_CONST(float, op->GetAttr("scale"));
  auto bias_ = BOOST_GET_CONST(float, op->GetAttr("bias"));
  auto bias_after_scale_ =
      BOOST_GET_CONST(bool, op->GetAttr("bias_after_scale"));
  auto data_type_ = op->Block()->FindVar(op->Input("X")[0])->GetDataType();

  // TODO(yaozhixin): support tensor as scale input
  if (abs(scale_ - 1.0) < 1e-06 && abs(bias_ - 0.0) < 1e-06) {
    auto new_node_identity = CreateBaseOp(
        graph, "popart_identity", {GetInputNode("X", node)}, node->outputs, {});
    return new_node_identity;
  } else {
    auto new_node_bias =
        CreateConst(graph, {}, {}, {{"value", std::vector<float>{bias_}},
                                    {"dims", std::vector<int64_t>{1}},
                                    {"dtype", ONNXDataType::FLOAT}});
    auto new_node_scale =
        CreateConst(graph, {}, {}, {{"value", std::vector<float>{scale_}},
                                    {"dims", std::vector<int64_t>{1}},
                                    {"dtype", ONNXDataType::FLOAT}});
    // convert to float32
    auto new_node_cast = CreateCast(graph, {GetInputNode("X", node)}, {},
                                    static_cast<int>(proto::VarType::FP32));

    ir::Node *result = nullptr;
    if (bias_after_scale_) {
      auto new_node_mul = CreateBaseOp(
          graph, "popart_mul",
          {new_node_cast->outputs[0], new_node_scale->outputs[0]}, {}, {});
      result = CreateBaseOp(
          graph, "popart_add",
          {new_node_mul->outputs[0], new_node_bias->outputs[0]}, {}, {});
    } else {
      auto new_node_add = CreateBaseOp(
          graph, "popart_add",
          {new_node_cast->outputs[0], new_node_bias->outputs[0]}, {}, {});
      result = CreateBaseOp(
          graph, "popart_mul",
          {new_node_add->outputs[0], new_node_scale->outputs[0]}, {}, {});
    }
    auto result_after_cast = CreateCast(graph, result->outputs, node->outputs,
                                        static_cast<int>(data_type_));
    return result_after_cast;
  }
}

ir::Node *cross_entropy2_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();
  auto ignoreIndex = BOOST_GET_CONST(int, op->GetAttr("ignore_index"));
  auto new_cast = CreateCast(graph, {GetInputNode("Label", node)}, {},
                             proto::VarType::INT32);
  return CreateBaseOp(graph, "popart_nllloss",
                      {GetInputNode("X", node), new_cast->outputs[0]},
                      {GetOutputNode("Y", node)},
                      {
                          {"ignoreIndex", ignoreIndex},
                      });
}

REGISTER_HANDLER(reduce_mean, reduce_mean_handler);
REGISTER_HANDLER(mean, mean_handler);
REGISTER_HANDLER(pow, pow_handler);
REGISTER_HANDLER(mul, mul_handler);
REGISTER_HANDLER(matmul, matmul_handler);
REGISTER_HANDLER(sum, sum_handler);
REGISTER_HANDLER(softmax, softmax_handler);
REGISTER_HANDLER(scale, scale_handler);
REGISTER_HANDLER(cross_entropy2, cross_entropy2_handler);

}  // namespace
}  // namespace ipu
}  // namespace framework
}  // namespace paddle
