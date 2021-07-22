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
#include <vector>

#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

std::shared_ptr<IpuBackend> IpuBackend::instance_ = nullptr;

popart::DataType toPopartType(proto::VarType::Type type) {
  switch (type) {
    case proto::VarType::UINT8:
      return popart::DataType::UINT8;
    case proto::VarType::INT8:
      return popart::DataType::INT8;
    case proto::VarType::INT16:
      return popart::DataType::INT16;
    case proto::VarType::INT32:
      return popart::DataType::INT32;
    case proto::VarType::INT64:
      return popart::DataType::INT64;
    case proto::VarType::BOOL:
      return popart::DataType::BOOL;
    case proto::VarType::FP32:
      return popart::DataType::FLOAT;
    case proto::VarType::FP16:
      return popart::DataType::FLOAT16;
    case proto::VarType::BF16:
      return popart::DataType::BFLOAT16;
    case proto::VarType::COMPLEX64:
      return popart::DataType::COMPLEX64;
    case proto::VarType::COMPLEX128:
      return popart::DataType::COMPLEX128;

    default:
      PADDLE_THROW(
          platform::errors::Unavailable("Unsupported Paddle var type."));
  }
}

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
          popart::DataType data_type = toPopartType(var_desc->GetDataType());
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

  for (const ir::Node* n : graph->Nodes()) {
    if (n->IsOp()) {
      auto* op_desc = n->Op();
      if (op_desc->Type() == "elementwise_add") {
        if (inputs_.size() != 2) {
          PADDLE_THROW(platform::errors::InvalidArgument("Invalid inputs."));
        }
        VLOG(1) << "found elementwise_add op";
        popart::TensorId lhs = inputs_[0];
        popart::TensorId rhs = inputs_[1];
        VLOG(1) << "popart add lhs tensor id = " << lhs;
        VLOG(1) << "popart add rhs tensor id = " << rhs;
        popart::TensorId result = builder_->aiOnnxOpset11().add({lhs, rhs});
        VLOG(1) << "popart add result tensor id = " << result;
        tensors_.emplace(fetch_list[0], result);
      } else {
        PADDLE_THROW(platform::errors::Unimplemented("Unimplemented."));
      }
    }
  }

  VLOG(1) << "-- fetch_list --";
  for (const auto& fetch_name : fetch_list) {
    VLOG(1) << fetch_name;
  }

  for (const auto& fetch_name : fetch_list) {
    auto tensor = tensors_.find(fetch_name);
    PADDLE_ENFORCE_NE(
        tensor, tensors_.end(),
        platform::errors::NotFound("output tensor %s does not exist.", fetch_name));

    VLOG(1) << "fetch_name= " << fetch_name;
    VLOG(1) << "popart output tensor id = " << tensor->second;
    builder_->addOutputTensor(tensor->second);
    outputs_.push_back(tensor->second);
  }

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

void IpuBackend::Run(const std::vector<const Tensor *> &inputs,
                     std::vector<Tensor *> &outputs) {
  // Prepare input tensor
  std::map<popart::TensorId, popart::IArray&> popart_inputs;
  std::map<popart::TensorId, popart::NDArrayWrapper<float>> input_wrappers;

  for (size_t i = 0; i < inputs.size(); i++) {
    auto tensor_id = inputs_[i];
    const Tensor* tensor = inputs[i];
    std::vector<int64_t> tensor_shape = builder_->getTensorShape(tensor_id);
    popart::NDArrayWrapper<float> data(const_cast<float*>(tensor->data<float>()), tensor_shape);
    VLOG(1) << "Preparing Input data for tensor " << tensor_id;
    input_wrappers.emplace(tensor_id, std::move(data));
    popart_inputs.emplace(tensor_id, input_wrappers.at(tensor_id));
  }

  // Prepare output tensor
  std::map<popart::TensorId, popart::IArray&> popart_anchors;
  std::map<popart::TensorId, popart::NDArrayWrapper<float>> anchor_wrappers;
  for(size_t i = 0; i < outputs.size(); i++) {
    auto tensor_id = outputs_[i];
    Tensor* tensor = outputs[i];
    std::vector<int64_t> tensor_shape = builder_->getTensorShape(tensor_id);
    popart::NDArrayWrapper<float> data(const_cast<float*>(tensor->data<float>()), tensor_shape);
    VLOG(1) << "Preparing Output data for tensor " << tensor_id;
    anchor_wrappers.emplace(tensor_id, std::move(data));
    popart_anchors.emplace(tensor_id, anchor_wrappers.at(tensor_id));
  }

  popart::StepIO stepio(popart_inputs, popart_anchors);

  VLOG(1) << "Running...";
  session_->run(stepio);
  VLOG(1) << "Running...done";
}

}  // namespace framework
}  // namespace paddle