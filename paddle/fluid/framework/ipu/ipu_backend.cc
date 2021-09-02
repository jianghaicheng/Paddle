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

#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/ipu/ipu_utils.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ipu {

std::shared_ptr<IpuBackend> instance_ = nullptr;

IpuBackend::IpuBackend() { compiler_ = std::make_shared<Compiler>(); }

IpuBackend::~IpuBackend() {
  // detach device
  if (curr_device_ != nullptr && curr_device_->isAttached()) {
    curr_device_->detach();
  }
}

std::shared_ptr<IpuBackend> IpuBackend::GetInstance() {
  if (!instance_) {
    instance_.reset(new IpuBackend());
  }
  return instance_;
}

// This api should only call from python, always return a new object
std::shared_ptr<IpuBackend> IpuBackend::GetNewInstance() {
  instance_.reset(new IpuBackend());
  return instance_;
}

void IpuBackend::Compile(ir::Graph* graph,
                         const std::vector<std::string>& feed_list,
                         const std::vector<std::string>& fetch_list) {
  VLOG(10) << "enter IpuBackend::Compile";
  compiler_->InitInputs(graph, feed_list);
  compiler_->LowerWeights(graph, scope_);
  compiler_->LowerBody(graph);
  compiler_->InitOutputs(fetch_list);
  VLOG(10) << "leave IpuBackend::Compile";
}

void IpuBackend::Run(const std::vector<const Tensor*>& inputs,
                     const std::vector<Tensor*>& outputs) {
  if (!is_prepared_) {
    Prepare();
    is_prepared_ = true;
  }

  std::map<popart::TensorId, popart::IArray&> popart_inputs;
  std::map<popart::TensorId, PaddleIArray> input_wrappers;
  auto input_tensors = compiler_->GetInputs();
  for (size_t i = 0; i < inputs.size(); i++) {
    auto tensor_id = input_tensors[i];
    auto tensor = const_cast<Tensor*>(inputs[i]);
    input_wrappers.emplace(tensor_id, PaddleIArray(tensor));
    popart_inputs.emplace(tensor_id, input_wrappers.at(tensor_id));
  }

  std::map<popart::TensorId, popart::IArray&> popart_anchors;
  std::map<popart::TensorId, PaddleIArray> anchor_wrappers;
  auto output_tensors = compiler_->GetOutputs();
  for (size_t i = 0; i < outputs.size(); i++) {
    auto tensor_id = output_tensors[i];
    auto tensor = const_cast<Tensor*>(outputs[i]);
    anchor_wrappers.emplace(tensor_id, PaddleIArray(tensor));
    popart_anchors.emplace(tensor_id, anchor_wrappers.at(tensor_id));
  }

  if (ipu_strategy_ != nullptr && ipu_strategy_->is_training) {
    VLOG(10) << "Update optimizer learning rate...";
    auto popart_optimizer = GetPopartOptimizer();
    auto &session = dynamic_cast<popart::TrainingSession &>(*session_);
    session.updateOptimizerFromHost(popart_optimizer.get());
  }

  popart::StepIO stepio(popart_inputs, popart_anchors);
  VLOG(10) << "Running...";
  session_->run(stepio);
  VLOG(10) << "Running...done";
}

void IpuBackend::Prepare() {
  VLOG(10) << "Get ModelProto ...\n";
  auto proto = compiler_->GetModelProto();
  VLOG(10) << "Save Model to file paddle_model.onnx ...\n";
  compiler_->SaveModelProto("paddle_model.onnx");
  VLOG(10) << "Constructing DataFlow\n";

  auto art = popart::AnchorReturnType("All");
  std::map<popart::TensorId, popart::AnchorReturnType> anchor_ids;
  for (const auto &id : compiler_->GetOutputs()) {
    anchor_ids.emplace(id, art);
  }
  auto dataFlow = popart::DataFlow(ipu_strategy_->batches_per_step, anchor_ids);

  PADDLE_ENFORCE_NOT_NULL(
      curr_device_,
      platform::errors::Unavailable("IPU device isn't attached, please call "
                                    "IpuBackend::AttachDevice(id) first."));

  if (ipu_strategy_ != nullptr && ipu_strategy_->is_training) {
    VLOG(10) << "Creating TrainingSession from Onnx Model...";
    auto popart_optimizer = GetPopartOptimizer();
    auto tensors = compiler_->GetTensors();
    auto it = tensors.find(optimizer_.loss);
    PADDLE_ENFORCE_NE(
        it, tensors.end(),
        paddle::platform::errors::InvalidArgument(
            "loss_id = %s doesn't exist in popart graph.", optimizer_.loss));
    session_ = popart::TrainingSession::createFromOnnxModel(
        proto, dataFlow, it->second, *popart_optimizer, curr_device_,
        popart::InputShapeInfo(), ipu_strategy_->popart_options_,
        popart::Patterns(popart::PatternsLevel::Default));
  } else {
    VLOG(10) << "Creating InferenceSession from Onnx Model...";
    session_ = popart::InferenceSession::createFromOnnxModel(
        proto, dataFlow, curr_device_, popart::InputShapeInfo(),
        ipu_strategy_->popart_options_,
        popart::Patterns(popart::PatternsLevel::Default));
  }
  VLOG(10) << "Creating session from Onnx Model...done";

  VLOG(10) << "Preparing session device...";
  session_->prepareDevice();
  VLOG(10) << "Preparing session device...done";

  VLOG(10) << "Copy weights from host to device...";
  session_->weightsFromHost();
  VLOG(10) << "Copy weights from host to device...done";
}

std::vector<int64_t> IpuBackend::GetTensorShape(const std::string& var_name) {
  auto oshape = compiler_->GetTensorShape(var_name);
  if (ipu_strategy_->batches_per_step != 1) {
    oshape.insert(oshape.begin(), ipu_strategy_->batches_per_step);
  }
  return oshape;
}

std::unique_ptr<popart::Optimizer> IpuBackend::GetPopartOptimizer() {
  // TODO(xiaobingw): change type to enum
  PADDLE_ENFORCE_NE(
      optimizer_.type, "",
      platform::errors::InvalidArgument("Optimizer type have not been set."));
  if (optimizer_.type == "sgd") {
    auto optimizer = std::make_unique<popart::SGD>(
        popart::OptimizerValue(GetLRFromScope(), false),
        popart::OptimizerValue(popart::SGD::getUnsetWeightDecay()),
        popart::OptimizerValue(popart::SGD::getUnsetMomentum()),
        popart::OptimizerValue(popart::SGD::getUnsetDampening()),
        popart::OptimizerValue(popart::SGD::getUnsetVelocityScaling()),
        popart::OptimizerValue(popart::SGD::getUnsetLossScaling()));
    return optimizer;
  } else if (optimizer_.type == "adam") {
    auto optimizer = std::make_unique<popart::Adam>(
        popart::OptimizerValue(GetLRFromScope(), false),
        popart::OptimizerValue(popart::Adam::getUnsetWeightDecay()),
        popart::OptimizerValue(GetOptimizerAttr("beta1"), false),
        popart::OptimizerValue(GetOptimizerAttr("beta2"), false),
        popart::OptimizerValue(GetOptimizerAttr("epsilon"), false),
        popart::OptimizerValue(popart::Adam::getUnsetLossScaling()),
        popart::AdamMode::Adam, popart::WeightDecayMode::Decay,
        popart::DataType::FLOAT, popart::DataType::FLOAT,
        popart::DataType::FLOAT);
    return optimizer;
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Optimizer %s is not implemented now.", optimizer_.type));
  }
}

float IpuBackend::GetOptimizerAttr(const std::string& attr,
                                   float default_value) {
  if (optimizer_.attrs.count(attr) == 0) {
    return default_value;
  }
  return optimizer_.attrs.at(attr);
}

void IpuBackend::SetOptimizerAttr(const std::string& attr, float value) {
  optimizer_.attrs[attr] = value;
}

float IpuBackend::GetLRFromScope() {
  auto lr_var = scope_->GetVar(optimizer_.lr_var_name);
  auto tensor = lr_var->Get<framework::LoDTensor>();

  PADDLE_ENFORCE_EQ(tensor.type(), framework::proto::VarType::FP32,
                    platform::errors::InvalidArgument(
                        "LR requiree float, but got (%s).", tensor.type()));

  return tensor.data<float>()[0];
}

void IpuBackend::SetIpuStrategy(const IpuStrategy& strategy) {
  ipu_strategy_ = &strategy;
}

size_t IpuBackend::GetNumDevices() {
  // IpuModel
  bool ipu_model = GetBoolEnv("POPLAR_IPUMODEL");
  if (ipu_model) return 1;
  // Real dev
  size_t num_devices =
      popart::DeviceManager::createDeviceManager().enumerateDevices().size();
  PADDLE_ENFORCE_GT(
      num_devices, 0,
      platform::errors::Unavailable(
          "Do not found any IPU devices, please make "
          "sure Poplar sdk is enabled or enable ENV \"POPLAR_IPUMODEL=1\""));
  return num_devices;
}

std::vector<int> IpuBackend::GetDeviceIds() {
  bool ipu_model = GetBoolEnv("POPLAR_IPUMODEL");
  if (ipu_model) {
    return {0};
  }
  std::vector<int> device_ids;
  auto devices =
      popart::DeviceManager::createDeviceManager().enumerateDevices();
  PADDLE_ENFORCE_GT(
      devices.size(), 0,
      platform::errors::Unavailable("Do not found any IPU devices, please make "
                                    "sure Poplar sdk is enabled."));

  for (auto device : devices) {
    device_ids.push_back(device->getId());
  }

  return device_ids;
}

Device IpuBackend::GetDevice(int id) {
  bool ipu_model = GetBoolEnv("POPLAR_IPUMODEL");
  if (ipu_model) {
    std::map<std::string, std::string> deviceOpts{{"numIPUs", "1 "}};
    curr_device_ =
        popart::DeviceManager::createDeviceManager().createIpuModelDevice(
            deviceOpts);
    Device device(*curr_device_.get());
    return device;
  }
  size_t num_devices = GetNumDevices();
  if (id < 0 || id >= num_devices) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "device id %d is invalid, number devices is %d", id, num_devices));
  }
  std::shared_ptr<popart::DeviceInfo> popart_device_info =
      popart::DeviceManager::createDeviceManager().getDevice(
          popart::SyncPattern::Full, id);
  Device device(*popart_device_info.get());
  return device;
}

void IpuBackend::AttachDevice(int id) {
  // trick here
  // Compiler ipu is not same as the runtime ipu.
  VLOG(10) << "comile ipu id = " << id;
  bool ipu_model = GetBoolEnv("POPLAR_IPUMODEL");
  if (ipu_model) {
    return;
  }
  curr_device_ =
      popart::DeviceManager::createDeviceManager().acquireAvailableDevice(
          UpperIpuNum());
  PADDLE_ENFORCE_NOT_NULL(
      curr_device_, platform::errors::Unavailable(
                        "Can't attach IPU, ipu_num = %d.", UpperIpuNum()));
}

bool IpuBackend::DeviceIsAttached() { return curr_device_ != nullptr; }

// num_ipus must be pow(2,n);
int IpuBackend::UpperIpuNum() {
  PADDLE_ENFORCE_GT(ipu_strategy_->num_ipus, 0,
                    platform::errors::Unavailable(
                        "The ipu num get is wrong, please make sure the "
                        "sharding or pipline parameter is right."));
  int i = 0;
  while (pow(2, i) < ipu_strategy_->num_ipus) {
    i++;
  }
  return pow(2, i);
}

}  // namespace ipu
}  // namespace framework
}  // namespace paddle
