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
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ipu {
namespace {

ir::Node *conv2d_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();
  auto op_desc = std::make_unique<framework::OpDesc>();
  op_desc->SetType("Conv");

  std::vector<std::string> inputs;
  inputs.push_back(op->Input("Input").front());
  inputs.push_back(op->Input("Filter").front());
  if (op->HasInput("Bias")) {
    if (!op->Input("Bias").empty()) {
      inputs.push_back(op->Input("Bias").front());
    }
  }
  op_desc->SetInput("__inputs__", inputs);
  std::vector<std::string> outputs;
  outputs.push_back(op->Output("Output").front());
  op_desc->SetOutput("__outputs__", outputs);

  auto dilations_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("dilations"));
  auto dilations = std::vector<int64_t>{dilations_.begin(), dilations_.end()};
  auto group_ = BOOST_GET_CONST(int, op->GetAttr("groups"));
  auto group = int64_t{group_};
  auto pads_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("paddings"));
  if (pads_.size() == 2) {
    pads_.push_back(pads_[0]);
    pads_.push_back(pads_[1]);
  }
  auto pads = std::vector<int64_t>{pads_.begin(), pads_.end()};
  auto stride_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("strides"));
  auto stride = std::vector<int64_t>{stride_.begin(), stride_.end()};
  op_desc->SetAttr("dilations", dilations);
  op_desc->SetAttr("group", group);
  op_desc->SetAttr("pads", pads);
  op_desc->SetAttr("strides", stride);

  op_desc->Flush();
  return graph->CreateOpNode(op_desc.get());
}

ir::Node *batch_norm_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();
  auto op_desc = std::make_unique<framework::OpDesc>();
  op_desc->SetType("BatchNormalization");
  std::vector<std::string> inputs;
  inputs.push_back(op->Input("X").front());
  inputs.push_back(op->Input("Scale").front());
  inputs.push_back(op->Input("Bias").front());
  inputs.push_back(op->Input("Mean").front());
  inputs.push_back(op->Input("Variance").front());
  // inputs.push_back(op->Input("MomentumTensor").front());
  op_desc->SetInput("__inputs__", inputs);
  std::vector<std::string> outputs;
  outputs.push_back(op->Output("Y").front());
  outputs.push_back(op->Output("MeanOut").front());
  outputs.push_back(op->Output("VarianceOut").front());
  outputs.push_back(op->Output("SavedMean").front());
  outputs.push_back(op->Output("SavedVariance").front());
  outputs.push_back(op->Output("ReserveSpace").front());
  op_desc->SetOutput("__outputs__", outputs);
  // attrs
  op_desc->SetAttr("momentum", BOOST_GET_CONST(float, op->GetAttr("momentum")));
  op_desc->SetAttr("epsilon", BOOST_GET_CONST(float, op->GetAttr("epsilon")));
  // op_desc->SetAttr("data_layout", BOOST_GET_CONST(string,
  // op->GetAttr("data_layout"));
  // op_desc->SetAttr("num_outputs", static_cast<int>(1));
  op_desc->Flush();
  return graph->CreateOpNode(op_desc.get());
}
ir::Node *pool2d_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();
  auto op_desc = std::make_unique<framework::OpDesc>();
  auto pool_type = BOOST_GET_CONST(std::string, op->GetAttr("pooling_type"));
  if (pool_type == "max") {
    op_desc->SetType("MaxPool");
  } else if (pool_type == "avg") {
    op_desc->SetType("AveragePool");
  }
  std::vector<std::string> inputs;
  inputs.push_back(op->Input("X").front());
  op_desc->SetInput("__inputs__", inputs);
  std::vector<std::string> outputs;
  outputs.push_back(op->Output("Out").front());
  op_desc->SetOutput("__outputs__", outputs);
  auto ksize = BOOST_GET_CONST(std::vector<int>, op->GetAttr("ksize"));

  std::vector<int64_t> kenel_shape_int64{ksize.begin(), ksize.end()};
  op_desc->SetAttr("kernel_shape", kenel_shape_int64);
  auto ceil_mode = BOOST_GET_CONST(bool, op->GetAttr("ceil_mode"));
  op_desc->SetAttr("ceil_mode", ceil_mode);

  // op_desc->SetAttr("dilations", {});
  auto pads = BOOST_GET_CONST(std::vector<int>, op->GetAttr("paddings"));
  std::vector<int64_t> paddings_int64{pads.begin(), pads.end()};
  if (paddings_int64.size() == 2) {
    paddings_int64.push_back(paddings_int64[0]);
    paddings_int64.push_back(paddings_int64[1]);
  }
  op_desc->SetAttr("paddings", paddings_int64);

  auto strides = BOOST_GET_CONST(std::vector<int>, op->GetAttr("strides"));
  std::vector<int64_t> strides_int64{strides.begin(), strides.end()};

  op_desc->SetAttr("strides", strides_int64);
  op_desc->SetAttr("count_include_pad", 0);
  op_desc->SetAttr("storage_order", 0);

  op_desc->Flush();
  return graph->CreateOpNode(op_desc.get());
}

ir::Node *group_norm_handler(ir::Graph *graph, ir::Node *node) {
  auto *op = node->Op();
  auto op_desc = std::make_unique<framework::OpDesc>();
  op_desc->SetType("GroupNorm");

  std::vector<std::string> inputs;
  inputs.push_back(op->Input("X").front());
  if (op->HasInput("Scale")) {
    if (!op->Input("Scale").empty()) {
      inputs.push_back(op->Input("Scale").front());
    }
  }
  if (op->HasInput("Bias")) {
    if (!op->Input("Bias").empty()) {
      inputs.push_back(op->Input("Bias").front());
    }
  }
  op_desc->SetInput("__inputs__", inputs);
  
  std::vector<std::string> outputs;
  outputs.push_back(op->Output("Y").front());
  outputs.push_back(op->Output("Mean").front());
  outputs.push_back(op->Output("Variance").front());
  op_desc->SetOutput("__outputs__", outputs);

  auto epsilon_ = BOOST_GET_CONST(float, op->GetAttr("epsilon"));
  op_desc->SetAttr("epsilon", epsilon_);
  auto groups_ = BOOST_GET_CONST(int, op->GetAttr("groups"));
  auto groups = int64_t{groups_};
  op_desc->SetAttr("groups", groups);

  op_desc->Flush();
  return graph->CreateOpNode(op_desc.get());
}

REGISTER_HANDLER(pool2d, pool2d_handler);
REGISTER_HANDLER(batch_norm, batch_norm_handler);
REGISTER_HANDLER(group_norm, group_norm_handler);
REGISTER_HANDLER(conv2d, conv2d_handler);

}  // namespace
}  // namespace ipu
}  // namespace framework
}  // namespace paddle
