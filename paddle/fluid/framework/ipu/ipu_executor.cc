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

#include "paddle/fluid/framework/ipu/ipu_executor.h"

namespace paddle {
namespace framework {
namespace ipu {

Executor::Executor() {}

Executor::~Executor() {}

void Executor::Prepare(const std::string &proto,
                       const std::map<std::string, popart::TensorId> &tensors,
                       const std::vector<popart::TensorId> &outputs,
                       std::shared_ptr<popart::DeviceInfo> device) {
  auto art = popart::AnchorReturnType("All");
  std::map<popart::TensorId, popart::AnchorReturnType> anchor_ids;
  for (const auto &id : outputs) {
    anchor_ids.emplace(id, art);
  }

  auto dataFlow = popart::DataFlow(ipu_strategy_->batches_per_step, anchor_ids);

  PADDLE_ENFORCE_NOT_NULL(device, platform::errors::Unavailable(
                                      "IPU device isn't attached, please call "
                                      "IpuBackend::AttachDevice(id) first."));

  if (ipu_strategy_ != nullptr && ipu_strategy_->is_training) {
    VLOG(10) << "Creating TrainingSession from Onnx Model...";
    auto popart_optimizer = GetPopartOptimizer(opt_info);

    auto it = tensors.find(opt_info.GetLoss());
    PADDLE_ENFORCE_NE(
        it, tensors.end(),
        paddle::platform::errors::InvalidArgument(
            "loss_id = %s doesn't exist in popart graph.", opt_info.GetLoss()));

    session_ = popart::TrainingSession::createFromOnnxModel(
        proto, dataFlow, it->second, *popart_optimizer, device,
        popart::InputShapeInfo(), ipu_strategy_->popart_options_,
        popart::Patterns(popart::PatternsLevel::Default));
  } else {
    VLOG(10) << "Creating InferenceSession from Onnx Model...";
    session_ = popart::InferenceSession::createFromOnnxModel(
        proto, dataFlow, device, popart::InputShapeInfo(),
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

void Executor::Run(const std::vector<popart::TensorId> &inputs_id,
                   const std::vector<const Tensor *> &inputs,
                   const std::vector<popart::TensorId> &outputs_id,
                   const std::vector<Tensor *> &outputs) {
  std::map<popart::TensorId, popart::IArray &> popart_inputs;
  std::map<popart::TensorId, PaddleIArray> input_wrappers;
  for (size_t i = 0; i < inputs.size(); i++) {
    auto tensor_id = inputs_id[i];
    auto tensor = const_cast<Tensor *>(inputs[i]);
    input_wrappers.emplace(tensor_id, PaddleIArray(tensor));
    popart_inputs.emplace(tensor_id, input_wrappers.at(tensor_id));
  }

  std::map<popart::TensorId, popart::IArray &> popart_anchors;
  std::map<popart::TensorId, PaddleIArray> anchor_wrappers;
  for (size_t i = 0; i < outputs.size(); i++) {
    auto tensor_id = outputs_id[i];
    auto tensor = const_cast<Tensor *>(outputs[i]);
    anchor_wrappers.emplace(tensor_id, PaddleIArray(tensor));
    popart_anchors.emplace(tensor_id, anchor_wrappers.at(tensor_id));
  }

  if (ipu_strategy_ != nullptr && ipu_strategy_->is_training) {
    VLOG(10) << "Update optimizer learning rate...";
    SetLR(GetLRFromScope());
    auto popart_optimizer = GetPopartOptimizer(opt_info);
    auto &session = dynamic_cast<popart::TrainingSession &>(*session_);
    session.updateOptimizerFromHost(popart_optimizer.get());
  }

  popart::StepIO stepio(popart_inputs, popart_anchors);
  VLOG(10) << "Running...";
  session_->run(stepio);
  VLOG(10) << "Running...done";
}

void Executor::SetOptimizerType(const std::string &type) {
  opt_info.SetType(type);
}

void Executor::SetLR(float lr_rate) { opt_info.SetLR(lr_rate); }

void Executor::SetOptimizerAttr(const std::string &attr, float value) {
  opt_info.SetAttr(attr, value);
}

void Executor::SetLoss(const std::string &loss) { opt_info.SetLoss(loss); }

void Executor::SetLRVarName(const std::string &name) {
  opt_info.SetLRVarName(name);
}

void Executor::SetIpuStrategy(const IpuStrategy &strategy) {
  ipu_strategy_ = &strategy;
}

void Executor::SetOutputsShape(
    const std::map<std::string, std::vector<int64_t>> &info) {
  for (const auto &pair : info) {
    outputs_shape_[pair.first] = pair.second;
  }
}

std::vector<int64_t> Executor::GetOutputShape(const std::string &fetch_name) {
  auto output_shape = outputs_shape_.at(fetch_name);
  if (ipu_strategy_->batches_per_step > 1) {
    output_shape.insert(output_shape.begin(), ipu_strategy_->batches_per_step);
  }
  return output_shape;
}

float Executor::GetLRFromScope() {
  auto lr_var = scope_->GetVar(opt_info.GetLRVarName());
  auto tensor = lr_var->Get<framework::LoDTensor>();

  PADDLE_ENFORCE_EQ(tensor.type(), framework::proto::VarType::FP32,
                    platform::errors::InvalidArgument(
                        "LR requiree float, but got (%s).", tensor.type()));

  return tensor.data<float>()[0];
}

}  // namespace ipu
}  // namespace framework
}  // namespace paddle
