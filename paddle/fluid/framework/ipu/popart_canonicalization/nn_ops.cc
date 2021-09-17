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

Node *conv2d_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto dilations_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("dilations"));
  auto dilations = std::vector<int64_t>{dilations_.begin(), dilations_.end()};
  auto group_ = BOOST_GET_CONST(int, op->GetAttr("groups"));
  auto pads_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("paddings"));
  if (pads_.size() == 2) {
    pads_.push_back(pads_[0]);
    pads_.push_back(pads_[1]);
  }
  auto pads = std::vector<int64_t>{pads_.begin(), pads_.end()};
  auto stride_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("strides"));
  auto stride = std::vector<int64_t>{stride_.begin(), stride_.end()};
  if (op->HasInput("Bias") && !op->Input("Bias").empty()) {
    return CreateConv(
        graph, node,
        {
            GetInputNode("Input", node), GetInputNode("Filter", node),
            GetInputNode("Bias", node),
        },
        node->outputs, dilations, group_, {}, pads, stride);
  } else {
    return CreateConv(
        graph, node,
        {
            GetInputNode("Input", node), GetInputNode("Filter", node),
        },
        node->outputs, dilations, group_, {}, pads, stride);
  }
}

Node *batch_norm_handler(Graph *graph, Node *node) {
  // TODO(alleng) differ from trainning & inference
  auto *op = node->Op();
  std::vector<Node *> inputs;
  inputs.push_back(GetInputNode("X", node));
  inputs.push_back(GetInputNode("Scale", node));
  inputs.push_back(GetInputNode("Bias", node));
  inputs.push_back(GetInputNode("Mean", node));
  inputs.push_back(GetInputNode("Variance", node));
  std::vector<Node *> outputs;
  outputs.push_back(GetOutputNode("Y", node));
  outputs.push_back(GetOutputNode("MeanOut", node));
  outputs.push_back(GetOutputNode("VarianceOut", node));
  outputs.push_back(GetOutputNode("SavedMean", node));
  outputs.push_back(GetOutputNode("SavedVariance", node));
  // outputs.push_back(GetOutputNode("ReserveSpace", node));
  auto momentum = BOOST_GET_CONST(float, op->GetAttr("momentum"));
  auto epsilon = BOOST_GET_CONST(float, op->GetAttr("epsilon"));
  // data_layout
  int64_t num_outputs = 1;
  return CreateBaseOp(graph, node, "popart_batchnormalization", inputs, outputs,
                      {
                          {"momentum", momentum},
                          {"epsilon", epsilon},
                          {"num_outputs", num_outputs},
                      });
}

Node *pool2d_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto pooling_type = BOOST_GET_CONST(std::string, op->GetAttr("pooling_type"));
  auto global_pooling = BOOST_GET_CONST(bool, op->GetAttr("global_pooling"));
  if (global_pooling) {
    if (pooling_type == "max") {
      return CreateBaseOp(graph, node, "popart_globalmaxpool", node->inputs,
                          node->outputs);
    } else if (pooling_type == "avg") {
      return CreateBaseOp(graph, node, "popart_globalaveragepool", node->inputs,
                          node->outputs);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "op pool2d with unkonwn pooling_type: %s", pooling_type));
    }
  }
  auto padding_algorithm =
      BOOST_GET_CONST(std::string, op->GetAttr("padding_algorithm"));
  if (padding_algorithm != "EXPLICIT") {
    // TODO(alleng) Fix this
    PADDLE_THROW(platform::errors::InvalidArgument(
        "op pool2d with unkonwn padding_algorithm: %s", padding_algorithm));
  }

  auto ksize = BOOST_GET_CONST(std::vector<int>, op->GetAttr("ksize"));
  auto kernel_shape = std::vector<int64_t>{ksize.begin(), ksize.end()};
  auto ceil_mode_ = BOOST_GET_CONST(bool, op->GetAttr("ceil_mode"));
  auto ceil_mode = int64_t(ceil_mode_ ? 1 : 0);
  auto paddings = BOOST_GET_CONST(std::vector<int>, op->GetAttr("paddings"));
  auto pads = std::vector<int64_t>{paddings.begin(), paddings.end()};
  if (pads.size() == 2) {
    pads.push_back(paddings[0]);
    pads.push_back(paddings[1]);
  }
  auto strides_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("strides"));
  auto strides = std::vector<int64_t>{strides_.begin(), strides_.end()};
  if (pooling_type == "max") {
    int64_t num_outputs = 1;
    auto dilations = std::vector<int64_t>{};
    int64_t storage_order = 0;
    return CreateBaseOp(graph, node, "popart_maxpool", node->inputs,
                        node->outputs, {
                                           {"num_outputs", num_outputs},
                                           {"kernel_shape", kernel_shape},
                                           {"ceil_mode", ceil_mode},
                                           {"dilations", dilations},
                                           {"pads", pads},
                                           {"storage_order", storage_order},
                                           {"strides", strides},
                                       });
  } else if (pooling_type == "avg") {
    int64_t count_include_pad = 0;
    return CreateBaseOp(graph, node, "popart_averagepool", node->inputs,
                        node->outputs,
                        {
                            {"kernel_shape", kernel_shape},
                            {"ceil_mode", ceil_mode},
                            {"count_include_pad", count_include_pad},
                            {"pads", pads},
                            {"strides", strides},
                        });
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "op pool2d with unkonwn pooling_type: %s", pooling_type));
  }
}

Node *group_norm_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto epsilon_ = BOOST_GET_CONST(float, op->GetAttr("epsilon"));
  auto groups_ = BOOST_GET_CONST(int, op->GetAttr("groups"));
  auto groups = int64_t{groups_};
  auto attrs_ = AttributeMap{{"epsilon", epsilon_}, {"num_groups", groups}};

  std::vector<Node *> inputs_ = {GetInputNode("X", node),
                                 GetInputNode("Scale", node),
                                 GetInputNode("Bias", node)};
  std::vector<Node *> outputs_ = {GetOutputNode("Y", node),
                                  GetOutputNode("Mean", node),
                                  GetOutputNode("Variance", node)};
  return CreateBaseOp(graph, node, "popart_groupnormalization", inputs_,
                      outputs_, attrs_);
}

Node *instance_norm_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto epsilon_ = BOOST_GET_CONST(float, op->GetAttr("epsilon"));
  auto attrs_ = AttributeMap{{"epsilon", epsilon_}};

  std::vector<Node *> inputs_ = {GetInputNode("X", node),
                                 GetInputNode("Scale", node),
                                 GetInputNode("Bias", node)};
  std::vector<Node *> outputs_ = {GetOutputNode("Y", node)};
  return CreateBaseOp(graph, node, "popart_instancenormalization", inputs_,
                      outputs_, attrs_);
}

Node *layer_norm_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto begin_norm_axis_ = BOOST_GET_CONST(int, op->GetAttr("begin_norm_axis"));
  auto input_shape_ = GetInputNode("X", node)->Var()->GetShape();

  std::vector<int64_t> norm_shape_{1, 1};
  for (int i = 0; i < input_shape_.size(); i++) {
    if (i < begin_norm_axis_) {
      norm_shape_[0] *= input_shape_[i];
    } else {
      norm_shape_[1] *= input_shape_[i];
    }
  }

  auto attrs1 = AttributeMap{
      {"value", norm_shape_},
      {"dims", std::vector<int64_t>{static_cast<int64_t>(norm_shape_.size())}},
      {"dtype", ONNXDataType::INT64}};
  auto reshape1_const =
      CreateBaseOp(graph, node, "popart_constant", {}, {}, attrs1);
  auto new_node_reshape1 = CreateBaseOp(
      graph, node, "popart_reshape",
      {GetInputNode("X", node), reshape1_const->outputs[0]}, {}, {});

  auto epsilon_ = BOOST_GET_CONST(float, op->GetAttr("epsilon"));
  int64_t groups_ = 1;
  auto groupnorm_attrs_ =
      AttributeMap{{"epsilon", epsilon_}, {"num_groups", groups_}};
  auto out_Y_ = MakeVarNode(graph, node);
  CreateBaseOp(
      graph, node, "popart_groupnormalization",
      {new_node_reshape1->outputs[0], GetInputNode("Scale", node),
       GetInputNode("Bias", node)},
      {out_Y_, GetOutputNode("Mean", node), GetOutputNode("Variance", node)},
      groupnorm_attrs_);

  auto attrs2 = AttributeMap{
      {"value", input_shape_},
      {"dims", std::vector<int64_t>{static_cast<int64_t>(input_shape_.size())}},
      {"dtype", ONNXDataType::INT64}};
  auto reshape2_const =
      CreateBaseOp(graph, node, "popart_constant", {}, {}, attrs2);
  auto new_node_reshape2 = CreateBaseOp(graph, node, "popart_reshape",
                                        {out_Y_, reshape2_const->outputs[0]},
                                        {GetOutputNode("Y", node)}, {});
  return new_node_reshape2;
}

Node *dropout_handler(Graph *graph, Node *node) {
  // TODO(yaozhixin): support training
  auto *op = node->Op();
  auto dropout_implementation_ =
      BOOST_GET_CONST(std::string, op->GetAttr("dropout_implementation"));
  auto dropout_prob_ = BOOST_GET_CONST(float, op->GetAttr("dropout_prob"));
  if (dropout_implementation_ == "upscale_in_train") {
    return CreateBaseOp(graph, node, "popart_identity",
                        {GetInputNode("X", node)}, {GetOutputNode("Out", node)},
                        {});
  } else if (dropout_implementation_ == "downgrade_in_infer") {
    auto scale = CreateConst(graph, node, {}, {},
                             {{"value", std::vector<float>{1 - dropout_prob_}},
                              {"dims", std::vector<int64_t>{1}},
                              {"dtype", ONNXDataType::FLOAT}});
    return CreateBaseOp(graph, node, "popart_mul",
                        {GetInputNode("X", node), scale->outputs[0]},
                        {GetOutputNode("Out", node)}, {});
  } else {
    PADDLE_THROW(
        platform::errors::InvalidArgument("Invalid dropout_implementation"));
  }
}

REGISTER_HANDLER(pool2d, pool2d_handler);
REGISTER_HANDLER(batch_norm, batch_norm_handler);
REGISTER_HANDLER(group_norm, group_norm_handler);
REGISTER_HANDLER(instance_norm, instance_norm_handler);
REGISTER_HANDLER(layer_norm, layer_norm_handler);
REGISTER_HANDLER(conv2d, conv2d_handler);
REGISTER_HANDLER(dropout, dropout_handler);

}  // namespace
}  // namespace ipu
}  // namespace framework
}  // namespace paddle
