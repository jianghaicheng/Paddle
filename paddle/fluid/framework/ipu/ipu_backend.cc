/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ipu/ipu_backend.h"

#include <algorithm>
#include <vector>

#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/names.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/session.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/stepio.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>

#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/ipu/ipu_utils.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

std::shared_ptr<IpuBackend> IpuBackend::instance_ = nullptr;

IpuBackend::IpuBackend() { builder_ = popart::Builder::create(); }

void IpuBackend::Compile(ir::Graph* graph,
                         const std::vector<std::string>& feed_list,
                         const std::vector<std::string>& fetch_list) {
  VLOG(1) << "-- feed_list --";
  for (const auto& feed_name : feed_list) {
    VLOG(1) << feed_name;

    for (const ir::Node* n : graph->Nodes()) {
      if (n->IsVar()) {
        auto* var_desc = n->Var();
        if (feed_name == var_desc->Name()) {
          // Get tensor_info from var_desc
          VLOG(1) << "feed_name= " << var_desc->Name();
          auto data_type = VarType2PopartType(var_desc->GetDataType());
          popart::TensorInfo input_info{data_type, var_desc->GetShape()};
          // Create popart tensor
          VLOG(1) << "popart input_info = " << input_info;
          popart::TensorId tensor_id = builder_->addInputTensor(input_info);
          VLOG(1) << "popart input tensor id = " << tensor_id;
          inputs_.push_back(tensor_id);
          tensors_.emplace(var_desc->Name(), tensor_id);
        }
      }
    }
  }

  LowerWeights(graph);
  LowerBody(graph);

  VLOG(1) << "-- fetch_list --";
  for (const auto& fetch_name : fetch_list) {
    VLOG(1) << fetch_name;
  }

  for (const auto& fetch_name : fetch_list) {
    auto tensor = tensors_.find(fetch_name);
    PADDLE_ENFORCE_NE(tensor, tensors_.end(),
                      platform::errors::NotFound(
                          "output tensor %s does not exist.", fetch_name));

    VLOG(1) << "fetch_name= " << fetch_name;
    VLOG(1) << "popart output tensor id = " << tensor->second;
    builder_->addOutputTensor(tensor->second);
    outputs_.push_back(tensor->second);
  }
}

void IpuBackend::Prepare() {
  VLOG(1) << "Save Model to file paddle_model.onnx ...\n";
  builder_->saveModelProto("paddle_model.onnx");

  VLOG(1) << "Get ModelProto ...\n";
  auto proto = builder_->getModelProto();

  VLOG(1) << "Constructing DataFlow\n";
  std::vector<popart::TensorId> anchor_ids;
  for (popart::TensorId item : outputs_) {
    anchor_ids.push_back(item);
  }
  auto dataFlow = popart::DataFlow(1, anchor_ids);

  std::map<std::string, std::string> deviceOpts{{"numIPUs", "1"}};
  auto ipuModelDevice =
      popart::DeviceManager::createDeviceManager().createIpuModelDevice(
          deviceOpts);
  // or acquireAvailableDevice();

  VLOG(1) << "Creating session from Onnx Model...";
  session_ = popart::InferenceSession::createFromOnnxModel(proto, dataFlow,
                                                           ipuModelDevice);
  VLOG(1) << "Creating session from Onnx Model...done";

  VLOG(1) << "Preparing session device...";
  session_->prepareDevice();
  VLOG(1) << "Preparing session device...done";
}

void IpuBackend::Run(const std::vector<const Tensor*>& inputs,
                     const std::vector<Tensor*>& outputs) {
  Prepare();

  std::map<popart::TensorId, popart::IArray&> popart_inputs;
  std::map<popart::TensorId, popart::NDArrayWrapper<float>> input_wrappers;
  // Prepare input tensor
  for (size_t i = 0; i < inputs.size(); i++) {
    auto tensor_id = inputs_[i];
    const Tensor* tensor = inputs[i];
    std::vector<int64_t> tensor_shape = builder_->getTensorShape(tensor_id);
    popart::NDArrayWrapper<float> data(
        const_cast<float*>(tensor->data<float>()), tensor_shape);
    VLOG(1) << "Preparing Input data for tensor " << tensor_id;
    input_wrappers.emplace(tensor_id, std::move(data));
    popart_inputs.emplace(tensor_id, input_wrappers.at(tensor_id));
  }
  // Prepare output tensor
  std::map<popart::TensorId, popart::IArray&> popart_anchors;
  std::map<popart::TensorId, popart::NDArrayWrapper<float>> anchor_wrappers;
  for (size_t i = 0; i < outputs.size(); i++) {
    auto tensor_id = outputs_[i];
    Tensor* tensor = outputs[i];
    std::vector<int64_t> tensor_shape = builder_->getTensorShape(tensor_id);
    popart::NDArrayWrapper<float> data(
        const_cast<float*>(tensor->data<float>()), tensor_shape);
    VLOG(1) << "Preparing Output data for tensor " << tensor_id;
    anchor_wrappers.emplace(tensor_id, std::move(data));
    popart_anchors.emplace(tensor_id, anchor_wrappers.at(tensor_id));
  }

  popart::StepIO stepio(popart_inputs, popart_anchors);

  VLOG(1) << "Running...";
  session_->run(stepio);
  VLOG(1) << "Running...done";
}

std::vector<std::string> IpuBackend::GetOpInputs(const OpDesc* op) {
  auto inputs_ = op->Input("__inputs__");
  std::vector<std::string> inputs;
  for (const auto& in : inputs_) {
    if (tensors_.find(in) != tensors_.end()) {
      inputs.push_back(tensors_[in]);
    } else {
      inputs.push_back(in);
    }
  }
  return inputs;
}

void IpuBackend::LowerWeights(const ir::Graph* graph) {
  PADDLE_ENFORCE_NOT_NULL(scope_,
                          platform::errors::PreconditionNotMet(
                              "You should call set_scope before LowerWeights"));

  // at this step, i think the graph doesn't contains optimizer
  // related states
  for (const auto* node : graph->Nodes()) {
    if (node->IsVar() && !node->IsCtrlVar() && node->Var()) {
      if (node->Var()->Persistable()) {
        auto var_name = node->Var()->Name();
        auto var = scope_->FindVar(var_name);
        if (var) {
          auto tensor = var->Get<framework::LoDTensor>();
          auto dtype = VarType2PopartType(tensor.type());
          auto shape = std::vector<int64_t>();
          for (size_t i = 0; i < tensor.dims().size(); ++i) {
            shape.push_back(tensor.dims().at(i));
          }
          popart::TensorInfo tensor_info(dtype, shape);
          popart::ConstVoidData const_data{tensor.data<void>(), tensor_info};
          popart::TensorId result =
              builder_->addInitializedInputTensor(const_data);
          tensors_.emplace(var_name, result);
        }
      }
    }
  }
}

void IpuBackend::LowerBody(const ir::Graph* graph) {
  auto nodes = TopologySortOperations(*graph);
  for (const auto* node : nodes) {
    auto* op = node->Op();
    auto op_type = op->Type();
    if (op_type == "RandomUniform") {
      auto outputs = op->Output("__outputs__");
      auto shape = BOOST_GET_CONST(std::vector<int64_t>, op->GetAttr("shape"));
      auto dtype = BOOST_GET_CONST(int, op->GetAttr("dtype"));
      auto high = BOOST_GET_CONST(float, op->GetAttr("high"));
      auto low = BOOST_GET_CONST(float, op->GetAttr("low"));
      popart::TensorId result =
          builder_->aiOnnxOpset11().randomuniform(shape, dtype, high, low);
      tensors_.emplace(outputs[0], result);
    } else if (op_type == "RandomNormal") {
      auto outputs = op->Output("__outputs__");
      auto shape = BOOST_GET_CONST(std::vector<int64_t>, op->GetAttr("shape"));
      auto dtype = BOOST_GET_CONST(int, op->GetAttr("dtype"));
      auto mean = BOOST_GET_CONST(float, op->GetAttr("mean"));
      auto scale = BOOST_GET_CONST(float, op->GetAttr("scale"));
      popart::TensorId result =
          builder_->aiOnnxOpset11().randomnormal(shape, dtype, mean, scale);
      tensors_.emplace(outputs[0], result);
    } else if (op_type == "ConstantOfShape") {
      // TODO(alleng) use RandomUniform for now
      auto outputs = op->Output("__outputs__");
      auto shape = BOOST_GET_CONST(std::vector<int64_t>, op->GetAttr("shape"));
      auto dtype = BOOST_GET_CONST(int, op->GetAttr("dtype"));
      auto high = 1.0f;
      auto low = 0.0f;
      popart::TensorId result =
          builder_->aiOnnxOpset11().randomuniform(shape, dtype, high, low);
      tensors_.emplace(outputs[0], result);
    } else if (op_type == "Add") {
      auto inputs = GetOpInputs(op);
      auto outputs = op->Output("__outputs__");
      popart::TensorId result = builder_->aiOnnxOpset11().add(inputs);
      tensors_.emplace(outputs[0], result);
    } else if (op_type == "Conv") {
      auto inputs = GetOpInputs(op);
      auto outputs = op->Output("__outputs__");
      auto dilations =
          BOOST_GET_CONST(std::vector<int64_t>, op->GetAttr("dilations"));
      auto group = BOOST_GET_CONST(int64_t, op->GetAttr("group"));
      auto pads = BOOST_GET_CONST(std::vector<int64_t>, op->GetAttr("pads"));
      auto strides =
          BOOST_GET_CONST(std::vector<int64_t>, op->GetAttr("strides"));
      popart::TensorId result = builder_->aiOnnxOpset11().conv(
          inputs, dilations, group, {}, pads, strides);
      tensors_.emplace(outputs[0], result);
    } else if (op_type == "ReduceMean") {
      auto inputs = GetOpInputs(op);
      auto outputs = op->Output("__outputs__");
      auto axes = BOOST_GET_CONST(std::vector<int64_t>, op->GetAttr("axes"));
      auto keepdims = BOOST_GET_CONST(int64_t, op->GetAttr("keepdims"));
      popart::TensorId result =
          builder_->aiOnnxOpset11().reducemean(inputs, axes, keepdims);
      tensors_.emplace(outputs[0], result);
    } else {
      PADDLE_THROW(platform::errors::Unimplemented("Unimplemented op type %s.",
                                                   op_type));
    }
  }
}

}  // namespace framework
}  // namespace paddle
