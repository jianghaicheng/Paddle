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

ir::Node *fill_constant_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();
  auto op_desc = std::make_unique<framework::OpDesc>();
  op_desc->SetType("popart_constant");
  if (op->HasInput("ShapeTensor") && !op->Input("ShapeTensor").empty()) {
    PADDLE_THROW(
        platform::errors::Unimplemented("op fill_constant with ShapeTensor"));
  }
  std::vector<std::string> outputs;
  outputs.push_back(op->Output("Out").front());
  op_desc->SetOutput("__outputs__", outputs);

  auto dtype_ = BOOST_GET_CONST(int, op->GetAttr("dtype"));
  auto dtype = VarType2OnnxDtype(dtype_);
  op_desc->SetAttr("dtype", dtype);
  auto dims = BOOST_GET_CONST(std::vector<int64_t>, op->GetAttr("shape"));
  op_desc->SetAttr("dims", dims);
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
  op_desc->SetAttr("value", value);

  op_desc->Flush();
  auto new_node = graph->CreateOpNode(op_desc.get());
  MoveNodeInputs(node, new_node);
  MoveNodeOutputs(node, new_node);
  return new_node;
}

ir::Node *gaussian_random_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();
  auto op_desc = std::make_unique<framework::OpDesc>();
  op_desc->SetType("popart_randomnormal");

  std::vector<std::string> outputs;
  outputs.push_back(op->Output("Out").front());
  op_desc->SetOutput("__outputs__", outputs);

  auto shape = BOOST_GET_CONST(std::vector<int64_t>, op->GetAttr("shape"));
  op_desc->SetAttr("shape", shape);
  auto dtype_ = BOOST_GET_CONST(int, op->GetAttr("dtype"));
  auto dtype = VarType2OnnxDtype(dtype_);
  op_desc->SetAttr("dtype", dtype);

  auto mean = BOOST_GET_CONST(float, op->GetAttr("mean"));
  op_desc->SetAttr("mean", mean);
  auto std = BOOST_GET_CONST(float, op->GetAttr("std"));
  op_desc->SetAttr("scale", std);
  // seed TODO

  op_desc->Flush();
  auto new_node = graph->CreateOpNode(op_desc.get());
  MoveNodeInputs(node, new_node);
  MoveNodeOutputs(node, new_node);
  return new_node;
}

ir::Node *uniform_random_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();
  auto op_desc = std::make_unique<framework::OpDesc>();
  op_desc->SetType("popart_randomuniform");

  std::vector<std::string> outputs;
  outputs.push_back(op->Output("Out").front());
  op_desc->SetOutput("__outputs__", outputs);

  auto shape = BOOST_GET_CONST(std::vector<int64_t>, op->GetAttr("shape"));
  op_desc->SetAttr("shape", shape);
  auto dtype_ = BOOST_GET_CONST(int, op->GetAttr("dtype"));
  auto dtype = VarType2OnnxDtype(dtype_);
  op_desc->SetAttr("dtype", dtype);
  auto max = BOOST_GET_CONST(float, op->GetAttr("max"));
  op_desc->SetAttr("high", max);
  auto min = BOOST_GET_CONST(float, op->GetAttr("min"));
  op_desc->SetAttr("low", min);
  // seed
  op_desc->Flush();
  auto new_node = graph->CreateOpNode(op_desc.get());
  MoveNodeInputs(node, new_node);
  MoveNodeOutputs(node, new_node);
  return new_node;
}

ir::Node *transpose_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();

  auto axis_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("axis"));
  std::vector<int64_t> perm(axis_.begin(), axis_.end());
  auto attrs = AttributeMap{{"perm", perm}};

  auto new_node_transpose = CreateBaseOp(graph, "popart_transpose",
                                         node->inputs, node->outputs, attrs);
  return new_node_transpose;
}

ir::Node *reshape_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();
  // TODO(yaozhixin) : Shape and ShapeTensor as inputs
  auto shape_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("shape"));
  std::vector<int64_t> shape(shape_.begin(), shape_.end());
  auto attrs = AttributeMap{
      {"value", shape},
      {"dims", std::vector<int64_t>{static_cast<int64_t>(shape.size())}},
      {"dtype", ONNXDataType::INT64}};
  auto new_node_const = CreateBaseOp(graph, "popart_constant", {}, {}, attrs);

  auto new_node_reshape =
      CreateBaseOp(graph, "popart_reshape",
                   {GetInputNode("X", node), new_node_const->outputs[0]},
                   {GetOutputNode("Out", node)}, {});
  return new_node_reshape;
}

ir::Node *gather_handler(ir::Graph *graph, ir::Node *node) {
  auto new_node_gather =
      CreateBaseOp(graph, "popart_gather",
                   {GetInputNode("X", node), GetInputNode("Index", node)},
                   {GetOutputNode("Out", node)}, {});
  return new_node_gather;
}

ir::Node *squeeze_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();
  auto axes_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("axes"));
  auto input_shape_ = op->Block()->FindVar(op->Input("X")[0])->GetShape();

  std::vector<int64_t> axes{axes_.begin(), axes_.end()};
  if (axes_.empty()) {
    for (int i = 0; i < input_shape_.size(); i++) {
      if (input_shape_[i] == 1) {
        axes.push_back(i);
      }
    }
  }
  auto new_node_squeeze =
      CreateBaseOp(graph, "popart_squeeze", {GetInputNode("X", node)},
                   {GetOutputNode("Out", node)}, {{"axes", axes}});

  return new_node_squeeze;
}

ir::Node *cast_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();
  auto otype = BOOST_GET_CONST(int, op->GetAttr("out_dtype"));
  auto new_node_cast = CreateCast(graph, node->inputs, node->outputs, otype);
  return new_node_cast;
}

ir::Node *lookup_table_handler(ir::Graph *graph, ir::Node *node) {
  auto new_node_squeeze =
      CreateBaseOp(graph, "popart_squeeze", {GetInputNode("Ids", node)}, {},
                   {{"axes", std::vector<int64_t>{-1}}});

  auto new_node_gather =
      CreateBaseOp(graph, "popart_gather",
                   {GetInputNode("W", node), new_node_squeeze->outputs[0]},
                   {GetOutputNode("Out", node)}, {});
  return new_node_gather;
}

ir::Node *unsqueeze_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();
  auto axes_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("axes"));
  std::vector<int64_t> axes{axes_.begin(), axes_.end()};
  auto new_node_unsqueeze =
      CreateBaseOp(graph, "popart_unsqueeze", {GetInputNode("X", node)},
                   node->outputs, {{"axes", axes}});

  return new_node_unsqueeze;
}

ir::Node *concat_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();
  // TODO(yaozhixin): support tensor as axis
  int64_t axis_{BOOST_GET_CONST(int, op->GetAttr("axis"))};

  auto new_node_concat = CreateBaseOp(graph, "popart_concat", node->inputs,
                                      node->outputs, {{"axis", axis_}});
  return new_node_concat;
}

ir::Node *stack_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();
  int64_t axis_{BOOST_GET_CONST(int, op->GetAttr("axis"))};
  std::vector<int64_t> axes_{axis_};

  std::vector<Node *> unsqueeze_outputs_{};
  for (auto input : node->inputs) {
    auto new_unsqueeze_node =
        CreateBaseOp(graph, "popart_unsqueeze", {input}, {}, {{"axes", axes_}});
    unsqueeze_outputs_.push_back(new_unsqueeze_node->outputs[0]);
    for (size_t i = 0; i < input->outputs.size(); ++i) {
      if (input->outputs[i] == node) {
        input->outputs[i] = new_unsqueeze_node;
        break;
      }
    }
  }
  auto new_node_concat =
      CreateBaseOp(graph, "popart_concat", unsqueeze_outputs_,
                   {GetOutputNode("Y", node)}, {{"axis", axis_}});
  return new_node_concat;
}

ir::Node *shape_handler(ir::Graph *graph, ir::Node *node) {
  auto new_node =
      CreateBaseOp(graph, "popart_shape", node->inputs, node->outputs);
  return new_node;
}

ir::Node *slice_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();
  Node *starts = nullptr;
  if (op->HasInput("StartsTensor") && !op->Input("StartsTensor").empty()) {
    starts = GetInputNode("StartsTensor", node);
  } else {
    auto starts_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("starts"));
    auto dim = int64_t(starts_.size());
    auto attr = MakeConstAttrMap<int>(starts_, {dim}, ONNXDataType::INT32);
    starts = CreateConst(graph, {}, {}, attr);
  }
  Node *ends = nullptr;
  if (op->HasInput("EndsTensor") && !op->Input("EndsTensor").empty()) {
    ends = GetInputNode("EndsTensor", node);
  } else {
    auto ends_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("ends"));
    auto dim = int64_t(ends_.size());
    auto attr = MakeConstAttrMap<int>(ends_, {dim}, ONNXDataType::INT32);
    ends = CreateConst(graph, {}, {}, attr);
  }
  Node *axes = nullptr;
  {
    auto axes_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("axes"));
    auto dim = int64_t(axes_.size());
    auto attr = MakeConstAttrMap<int>(axes_, {dim}, ONNXDataType::INT32);
    axes = CreateConst(graph, {}, {}, attr);
  }
  Node *steps = nullptr;
  {
    auto size =
        BOOST_GET_CONST(std::vector<int64_t>, ends->Op()->GetAttr("dims"))
            .front();
    auto steps_ = std::vector<int>(size, 1);
    auto dim = int64_t(size);
    auto attr = MakeConstAttrMap<int>(steps_, {dim}, ONNXDataType::INT32);
    steps = CreateConst(graph, {}, {}, attr);
  }
  auto new_node =
      CreateBaseOp(graph, "popart_slice",
                   {GetInputNode("Input", node), starts->outputs[0],
                    ends->outputs[0], axes->outputs[0], steps->outputs[0]},
                   node->outputs);
  return new_node;
}

ir::Node *expand_handler(ir::Graph *graph, ir::Node *node) {
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
    // TODO(alleng) using cast will fail at runtime
    expand_times = CreateCast(graph, {GetInputNode("ExpandTimes", node)}, {},
                              proto::VarType::INT64);
  } else {
    auto expand_times_i32 =
        BOOST_GET_CONST(std::vector<int>, op->GetAttr("expand_times"));
    auto expand_times_ =
        std::vector<int64_t>{expand_times_i32.begin(), expand_times_i32.end()};
    auto dim = int64_t(expand_times_.size());
    auto attr =
        MakeConstAttrMap<int64_t>(expand_times_, {dim}, ONNXDataType::INT64);
    expand_times = CreateConst(graph, {}, {}, attr);
  }
  auto new_node = CreateBaseOp(
      graph, "popart_tile", {GetInputNode("X", node), expand_times->outputs[0]},
      node->outputs);
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
