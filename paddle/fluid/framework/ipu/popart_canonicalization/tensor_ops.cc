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

Node *fill_constant_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  if (op->HasInput("ShapeTensor") && !op->Input("ShapeTensor").empty()) {
    PADDLE_THROW(
        platform::errors::Unimplemented("op fill_constant with ShapeTensor"));
  }
  auto dtype_ = BOOST_GET_CONST(int, op->GetAttr("dtype"));
  auto dtype = VarType2OnnxDtype(dtype_);
  auto dims = BOOST_GET_CONST(std::vector<int64_t>, op->GetAttr("shape"));
  auto value_ = BOOST_GET_CONST(float, op->GetAttr("value"));
  size_t size = 1;
  for (auto &dim : dims) {
    size *= dim;
  }
  Attribute value;
  switch (dtype_) {
    case proto::VarType::FP32:
      value = std::vector<float>(size, value_);
      break;
    case proto::VarType::FP64:
      value = std::vector<double>(size, value_);
      break;
    case proto::VarType::INT32:
      value = std::vector<int>(size, value_);
      break;
    case proto::VarType::INT64:
      value = std::vector<int64_t>(size, value_);
      break;
    default:
      PADDLE_THROW(
          platform::errors::Unimplemented("fill_constant dtype: %d", dtype_));
  }
  return CreateConst(graph, node, node->inputs, node->outputs,
                     AttributeMap{
                         {"value", value}, {"dims", dims}, {"dtype", dtype},
                     });
}

Node *gaussian_random_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto shape = BOOST_GET_CONST(std::vector<int64_t>, op->GetAttr("shape"));
  auto dtype_ = BOOST_GET_CONST(int, op->GetAttr("dtype"));
  auto dtype = VarType2OnnxDtype(dtype_);
  auto mean = BOOST_GET_CONST(float, op->GetAttr("mean"));
  auto scale = BOOST_GET_CONST(float, op->GetAttr("std"));
  // TODO(alleng) seed not work
  auto seed_ = BOOST_GET_CONST(int, op->GetAttr("seed"));
  auto seed = static_cast<float>(seed_);
  return CreateBaseOp(graph, node, "popart_randomnormal", node->inputs,
                      node->outputs, {
                                         {"shape", shape},
                                         {"dtype", dtype},
                                         {"mean", mean},
                                         {"scale", scale},
                                         {"seed", seed},
                                     });
}

Node *uniform_random_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto shape = BOOST_GET_CONST(std::vector<int64_t>, op->GetAttr("shape"));
  auto dtype_ = BOOST_GET_CONST(int, op->GetAttr("dtype"));
  auto dtype = VarType2OnnxDtype(dtype_);
  auto high = BOOST_GET_CONST(float, op->GetAttr("max"));
  auto low = BOOST_GET_CONST(float, op->GetAttr("min"));
  // TODO(alleng) seed not work
  auto seed_ = BOOST_GET_CONST(int, op->GetAttr("seed"));
  auto seed = static_cast<float>(seed_);
  return CreateBaseOp(graph, node, "popart_randomuniform", node->inputs,
                      node->outputs, {
                                         {"shape", shape},
                                         {"dtype", dtype},
                                         {"high", high},
                                         {"low", low},
                                         {"seed", seed},
                                     });
}

Node *transpose_handler(Graph *graph, Node *node) {
  auto *op = node->Op();

  auto axis_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("axis"));
  std::vector<int64_t> perm(axis_.begin(), axis_.end());
  auto attrs = AttributeMap{{"perm", perm}};

  auto new_node_transpose =
      CreateBaseOp(graph, node, "popart_transpose", node->inputs,
                   {GetOutputVarNode("Out", node)}, attrs);
  return new_node_transpose;
}

Node *reshape_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto shape_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("shape"));
  std::vector<int64_t> shape(shape_.begin(), shape_.end());
  auto attrs = AttributeMap{
      {"value", shape},
      {"dims", std::vector<int64_t>{static_cast<int64_t>(shape.size())}},
      {"dtype", ONNXDataType::INT64}};
  auto new_node_const =
      CreateBaseOp(graph, node, "popart_constant", {}, {}, attrs);

  auto new_node_reshape =
      CreateBaseOp(graph, node, "popart_reshape",
                   {GetInputVarNode("X", node), new_node_const->outputs[0]},
                   {GetOutputVarNode("Out", node)}, {});
  return new_node_reshape;
}

Node *gather_handler(Graph *graph, Node *node) {
  auto new_node_gather =
      CreateBaseOp(graph, node, "popart_gather",
                   {GetInputVarNode("X", node), GetInputVarNode("Index", node)},
                   {GetOutputVarNode("Out", node)}, {});
  return new_node_gather;
}

Node *squeeze_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto axes_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("axes"));
  auto input_shape_ = GetInputVarNode("X", node)->Var()->GetShape();

  std::vector<int64_t> axes{axes_.begin(), axes_.end()};
  if (axes_.empty()) {
    for (int i = 0; i < input_shape_.size(); i++) {
      if (input_shape_[i] == 1) {
        axes.push_back(i);
      }
    }
  }
  auto new_node_squeeze =
      CreateBaseOp(graph, node, "popart_squeeze", {GetInputVarNode("X", node)},
                   {GetOutputVarNode("Out", node)}, {{"axes", axes}});

  return new_node_squeeze;
}

Node *cast_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto otype = BOOST_GET_CONST(int, op->GetAttr("out_dtype"));
  auto new_node_cast =
      CreateCast(graph, node, node->inputs, node->outputs, otype);
  return new_node_cast;
}

Node *lookup_table_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto padding_idx_ = BOOST_GET_CONST(int64_t, op->GetAttr("padding_idx"));
  auto w_shape_ = GetInputVarNode("W", node)->Var()->GetShape();
  auto table_size_ = w_shape_[0];
  auto emb_size_ = w_shape_[1];

  Node *w_node;
  if (padding_idx_ >= 0 && padding_idx_ < table_size_) {
    std::vector<float> const_value_(emb_size_, 0);
    std::vector<int64_t> const_shape_{1, emb_size_};
    auto concat_const =
        CreateConst(graph, node, {}, {}, {{"value", const_value_},
                                          {"dims", const_shape_},
                                          {"dtype", ONNXDataType::FLOAT}});
    auto axes =
        CreateConst(graph, node, {}, {}, {{"value", std::vector<int64_t>{0}},
                                          {"dims", std::vector<int64_t>{1}},
                                          {"dtype", ONNXDataType::INT64}});
    auto step =
        CreateConst(graph, node, {}, {}, {{"value", std::vector<int64_t>{1}},
                                          {"dims", std::vector<int64_t>{1}},
                                          {"dtype", ONNXDataType::INT64}});

    auto left_start =
        CreateConst(graph, node, {}, {}, {{"value", std::vector<int64_t>{0}},
                                          {"dims", std::vector<int64_t>{1}},
                                          {"dtype", ONNXDataType::INT64}});
    auto left_end = CreateConst(graph, node, {}, {},
                                {{"value", std::vector<int64_t>{padding_idx_}},
                                 {"dims", std::vector<int64_t>{1}},
                                 {"dtype", ONNXDataType::INT64}});

    auto right_start = CreateConst(
        graph, node, {}, {}, {{"value", std::vector<int64_t>{padding_idx_ + 1}},
                              {"dims", std::vector<int64_t>{1}},
                              {"dtype", ONNXDataType::INT64}});
    auto right_end = CreateConst(graph, node, {}, {},
                                 {{"value", std::vector<int64_t>{table_size_}},
                                  {"dims", std::vector<int64_t>{1}},
                                  {"dtype", ONNXDataType::INT64}});

    auto left_slice =
        CreateBaseOp(graph, node, "popart_slice",
                     {GetInputVarNode("W", node), left_start->outputs[0],
                      left_end->outputs[0], axes->outputs[0], step->outputs[0]},
                     {}, {});
    auto right_slice = CreateBaseOp(
        graph, node, "popart_slice",
        {GetInputVarNode("W", node), right_start->outputs[0],
         right_end->outputs[0], axes->outputs[0], step->outputs[0]},
        {}, {});

    if (padding_idx_ == 0) {
      w_node = CreateBaseOp(graph, node, "popart_concat",
                            {concat_const->outputs[0], right_slice->outputs[0]},
                            {}, {{"axis", int64_t(0)}});
      ClearNode(left_start);
      ClearNode(left_end);
      ClearNode(left_slice);
    } else if (padding_idx_ == table_size_ - 1) {
      w_node = CreateBaseOp(graph, node, "popart_concat",
                            {left_slice->outputs[0], concat_const->outputs[0]},
                            {}, {{"axis", int64_t{0}}});
      ClearNode(right_start);
      ClearNode(right_end);
      ClearNode(right_slice);
    } else {
      w_node = CreateBaseOp(graph, node, "popart_concat",
                            {left_slice->outputs[0], concat_const->outputs[0],
                             right_slice->outputs[0]},
                            {}, {{"axis", int64_t{0}}});
    }
    w_node = w_node->outputs[0];
  } else {
    w_node = GetInputVarNode("W", node);
  }

  auto squeeze = CreateBaseOp(graph, node, "popart_squeeze",
                              {GetInputVarNode("Ids", node)}, {},
                              {{"axes", std::vector<int64_t>{-1}}});

  auto gather =
      CreateBaseOp(graph, node, "popart_gather", {w_node, squeeze->outputs[0]},
                   {GetOutputVarNode("Out", node)}, {});
  return gather;
}

Node *unsqueeze_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto axes_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("axes"));
  std::vector<int64_t> axes{axes_.begin(), axes_.end()};
  auto new_node_unsqueeze = CreateBaseOp(
      graph, node, "popart_unsqueeze", {GetInputVarNode("X", node)},
      {GetOutputVarNode("Out", node)}, {{"axes", axes}});

  return new_node_unsqueeze;
}

Node *concat_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  int64_t axis_{BOOST_GET_CONST(int, op->GetAttr("axis"))};

  auto new_node_concat =
      CreateBaseOp(graph, node, "popart_concat", node->inputs, node->outputs,
                   {{"axis", axis_}});
  return new_node_concat;
}

Node *stack_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  int64_t axis_{BOOST_GET_CONST(int, op->GetAttr("axis"))};
  std::vector<int64_t> axes_{axis_};

  std::vector<Node *> unsqueeze_outputs_{};
  for (auto input : node->inputs) {
    auto new_unsqueeze_node = CreateBaseOp(graph, node, "popart_unsqueeze",
                                           {input}, {}, {{"axes", axes_}});
    unsqueeze_outputs_.push_back(new_unsqueeze_node->outputs[0]);
    for (size_t i = 0; i < input->outputs.size(); ++i) {
      if (input->outputs[i] == node) {
        input->outputs[i] = new_unsqueeze_node;
        break;
      }
    }
  }
  auto new_node_concat =
      CreateBaseOp(graph, node, "popart_concat", unsqueeze_outputs_,
                   {GetOutputVarNode("Y", node)}, {{"axis", axis_}});
  return new_node_concat;
}

Node *shape_handler(Graph *graph, Node *node) {
  auto new_node =
      CreateBaseOp(graph, node, "popart_shape", node->inputs, node->outputs);
  return new_node;
}

Node *slice_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  Node *starts = nullptr;
  if (op->HasInput("StartsTensor") && !op->Input("StartsTensor").empty()) {
    starts = GetInputVarNode("StartsTensor", node);
  } else {
    auto starts_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("starts"));
    auto dim = int64_t(starts_.size());
    auto attr = MakeConstAttrMap<int>(starts_, {dim}, ONNXDataType::INT32);
    starts = CreateConst(graph, node, {}, {}, attr);
    starts = starts->outputs[0];
  }
  Node *ends = nullptr;
  if (op->HasInput("EndsTensor") && !op->Input("EndsTensor").empty()) {
    ends = GetInputVarNode("EndsTensor", node);
  } else {
    auto ends_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("ends"));
    auto dim = int64_t(ends_.size());
    auto attr = MakeConstAttrMap<int>(ends_, {dim}, ONNXDataType::INT32);
    ends = CreateConst(graph, node, {}, {}, attr);
    ends = ends->outputs[0];
  }
  Node *axes = nullptr;
  {
    auto axes_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("axes"));
    auto dim = int64_t(axes_.size());
    auto attr = MakeConstAttrMap<int>(axes_, {dim}, ONNXDataType::INT32);
    axes = CreateConst(graph, node, {}, {}, attr);
  }
  auto new_node = CreateBaseOp(
      graph, node, "popart_slice",
      {GetInputVarNode("Input", node), starts, ends, axes->outputs[0]},
      node->outputs);
  return new_node;
}

Node *expand_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  // TODO(alleng) work with expand_times_tensor
  if (op->HasInput("expand_times_tensor") &&
      !op->Input("expand_times_tensor").empty()) {
    PADDLE_THROW(
        platform::errors::Unimplemented("Expand op with expand_times_tensor"));
  }

  Node *expand_times = nullptr;
  if (op->HasInput("ExpandTimes") && !op->Input("ExpandTimes").empty()) {
    // cast to int64
    expand_times =
        CreateCast(graph, node, {GetInputVarNode("ExpandTimes", node)}, {},
                   proto::VarType::INT64);
  } else {
    auto expand_times_i32 =
        BOOST_GET_CONST(std::vector<int>, op->GetAttr("expand_times"));
    auto expand_times_ =
        std::vector<int64_t>{expand_times_i32.begin(), expand_times_i32.end()};
    auto dim = int64_t(expand_times_.size());
    auto attr =
        MakeConstAttrMap<int64_t>(expand_times_, {dim}, ONNXDataType::INT64);
    expand_times = CreateConst(graph, node, {}, {}, attr);
  }
  auto new_node = CreateBaseOp(
      graph, node, "popart_tile",
      {GetInputVarNode("X", node), expand_times->outputs[0]}, node->outputs);
  return new_node;
}

REGISTER_HANDLER(fill_constant, fill_constant_handler);
REGISTER_HANDLER(gaussian_random, gaussian_random_handler);
REGISTER_HANDLER(uniform_random, uniform_random_handler);
REGISTER_HANDLER(transpose2, transpose_handler);
REGISTER_HANDLER(reshape2, reshape_handler);
REGISTER_HANDLER(gather, gather_handler);
REGISTER_HANDLER(squeeze2, squeeze_handler);
REGISTER_HANDLER(cast, cast_handler);
REGISTER_HANDLER(lookup_table, lookup_table_handler);
REGISTER_HANDLER(unsqueeze2, unsqueeze_handler);
REGISTER_HANDLER(concat, concat_handler);
REGISTER_HANDLER(stack, stack_handler);
REGISTER_HANDLER(shape, shape_handler);
REGISTER_HANDLER(slice, slice_handler);
REGISTER_HANDLER(expand, expand_handler);

}  // namespace
}  // namespace ipu
}  // namespace framework
}  // namespace paddle
