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

std::unique_ptr<popart::Optimizer> IpuBackend::GetPopartOptimizer() {
  // TODO(xiaobingw): change type_ to enum
  PADDLE_ENFORCE_NE(
      optimizer_.type_, "",
      platform::errors::InvalidArgument("Optimizer type have not been set."));
  if (optimizer_.type_ == "sgd") {
    auto optimizer = std::make_unique<popart::SGD>(
        popart::OptimizerValue(GetLRFromScope(), false),
        popart::OptimizerValue(popart::SGD::getUnsetWeightDecay()),
        popart::OptimizerValue(popart::SGD::getUnsetMomentum()),
        popart::OptimizerValue(popart::SGD::getUnsetDampening()),
        popart::OptimizerValue(popart::SGD::getUnsetVelocityScaling()),
        popart::OptimizerValue(popart::SGD::getUnsetLossScaling()));
    return optimizer;
  } else if (optimizer_.type_ == "adam") {
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
        "Optimizer %s is not implemented now.", optimizer_.type_));
  }
}

void IpuBackend::Prepare() {
  VLOG(1) << "Get ModelProto ...\n";
  auto proto = builder_->getModelProto();

  // for onnx graph debug
  // std::ofstream onnxfile("paddle_model_no_check.onnx",
  // std::ios_base::binary);
  // onnxfile.write(proto.data(), proto.size());
  // onnxfile.close();

  VLOG(1) << "Save Model to file paddle_model.onnx ...\n";
  builder_->saveModelProto("paddle_model.onnx");

  VLOG(1) << "Constructing DataFlow\n";
  std::vector<popart::TensorId> anchor_ids;
  for (popart::TensorId item : outputs_) {
    anchor_ids.push_back(item);
  }
  auto dataFlow = popart::DataFlow(1, anchor_ids);

  PADDLE_ENFORCE_NOT_NULL(
      curr_device_,
      platform::errors::Unavailable("IPU device isn't attached, please call "
                                    "IpuBackend::AttachDevice(id) first."));

  if (ipu_strategy_ != nullptr && ipu_strategy_->is_training_) {
    VLOG(1) << "Creating TrainingSession from Onnx Model...";
    auto popart_optimizer = GetPopartOptimizer();
    auto it = tensors_.find(optimizer_.loss_);
    PADDLE_ENFORCE_NE(
        it, tensors_.end(),
        paddle::platform::errors::InvalidArgument(
            "loss_id = %s doesn't exist in popart graph.", optimizer_.loss_));
    session_ = popart::TrainingSession::createFromOnnxModel(
        proto, dataFlow, it->second, *popart_optimizer, curr_device_,
        popart::InputShapeInfo(), ipu_strategy_->popart_options_,
        popart::Patterns(popart::PatternsLevel::Default));
  } else {
    VLOG(1) << "Creating InferenceSession from Onnx Model...";
    session_ = popart::InferenceSession::createFromOnnxModel(
        proto, dataFlow, curr_device_, popart::InputShapeInfo(),
        ipu_strategy_->popart_options_,
        popart::Patterns(popart::PatternsLevel::Default));
  }
  VLOG(1) << "Creating session from Onnx Model...done";

  VLOG(1) << "Preparing session device...";
  session_->prepareDevice();
  VLOG(1) << "Preparing session device...done";

  VLOG(1) << "Copy weights from host to device...";
  session_->weightsFromHost();
  VLOG(1) << "Copy weights from host to device...done";
}

void IpuBackend::Run(const std::vector<const Tensor*>& inputs,
                     const std::vector<Tensor*>& outputs) {
  if (!is_prepared_) {
    Prepare();
    is_prepared_ = true;
  }

  std::map<popart::TensorId, popart::IArray&> popart_inputs;
  std::map<popart::TensorId, PaddleIArray> input_wrappers;
  for (size_t i = 0; i < inputs.size(); i++) {
    auto tensor_id = inputs_[i];
    auto tensor = const_cast<Tensor*>(inputs[i]);
    input_wrappers.emplace(tensor_id, PaddleIArray(tensor));
    popart_inputs.emplace(tensor_id, input_wrappers.at(tensor_id));
  }

  std::map<popart::TensorId, popart::IArray&> popart_anchors;
  std::map<popart::TensorId, PaddleIArray> anchor_wrappers;
  for (size_t i = 0; i < outputs.size(); i++) {
    auto tensor_id = outputs_[i];
    auto tensor = const_cast<Tensor*>(outputs[i]);
    anchor_wrappers.emplace(tensor_id, PaddleIArray(tensor));
    popart_anchors.emplace(tensor_id, anchor_wrappers.at(tensor_id));
  }

  if (ipu_strategy_ != nullptr && ipu_strategy_->is_training_) {
    VLOG(1) << "Update optimizer learning rate...";
    auto popart_optimizer = GetPopartOptimizer();
    auto session = dynamic_cast<popart::TrainingSession*>(session_.get());
    session->updateOptimizerFromHost(popart_optimizer.get());
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
  VLOG(10) << "enter IpuBackend::LowerBody";
  auto nodes = TopologySortOperations(*graph);
  for (const auto* node : nodes) {
    auto* op = node->Op();
    auto op_type = op->Type();
    VLOG(10) << "Lowering Node: " << node->Name() << " Op: " << op_type;
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
    } else if (op_type == "Constant") {
      auto outputs = op->Output("__outputs__");
      auto dims = BOOST_GET_CONST(std::vector<int64_t>, op->GetAttr("dims"));
      auto dtype_ = BOOST_GET_CONST(int, op->GetAttr("dtype"));
      auto dtype = OnnxDtype2PopartType(dtype_);
      popart::TensorInfo tensor_info{dtype, dims};
      auto value_attr = op->GetAttr("value");
      auto const_data = std::unique_ptr<popart::ConstVoidData>{};
      // Attribute only support vector of int,int64_t,float,double,bool
      // if need store other data type, we should try another way
      switch (dtype) {
        case popart::DataType::FLOAT:
          const_data.reset(new popart::ConstVoidData(
              BOOST_GET_CONST(std::vector<float>, value_attr).data(),
              tensor_info));
          break;
        case popart::DataType::INT32:
          const_data.reset(new popart::ConstVoidData(
              BOOST_GET_CONST(std::vector<int>, value_attr).data(),
              tensor_info));
          break;
        case popart::DataType::DOUBLE:
          const_data.reset(new popart::ConstVoidData(
              BOOST_GET_CONST(std::vector<double>, value_attr).data(),
              tensor_info));
          break;
        case popart::DataType::INT64:
          const_data.reset(new popart::ConstVoidData(
              BOOST_GET_CONST(std::vector<int64_t>, value_attr).data(),
              tensor_info));
          break;
        default:
          PADDLE_THROW(
              platform::errors::Unimplemented("popart::DataType %d", dtype));
      }
      popart::TensorId result = builder_->aiOnnxOpset11().constant(*const_data);
      tensors_.emplace(outputs[0], result);
    } else if (op_type == "Add") {
      auto inputs = GetOpInputs(op);
      auto outputs = op->Output("__outputs__");
      popart::TensorId result = builder_->aiOnnxOpset11().add(inputs);
      tensors_.emplace(outputs[0], result);
    } else if (op_type == "Mul") {
      auto inputs = GetOpInputs(op);
      auto outputs = op->Output("__outputs__");
      popart::TensorId result = builder_->aiOnnxOpset11().mul(inputs);
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
    } else if (op_type == "Equal") {
      auto inputs = GetOpInputs(op);
      auto outputs = op->Output("__outputs__");
      popart::TensorId result = builder_->aiOnnxOpset11().equal(inputs);
      tensors_.emplace(outputs[0], result);
    } else if (op_type == "MatMul") {
      auto inputs = GetOpInputs(op);
      auto outputs = op->Output("__outputs__");
      popart::TensorId result = builder_->aiOnnxOpset11().matmul(inputs);
      tensors_.emplace(outputs[0], result);
    } else if (op_type == "ReduceMean") {
      auto inputs = GetOpInputs(op);
      auto outputs = op->Output("__outputs__");
      auto axes = nonstd::optional<std::vector<int64_t>>();
      if (op->HasAttr("axes")) {
        axes = BOOST_GET_CONST(std::vector<int64_t>, op->GetAttr("axes"));
      }
      auto keepdims = BOOST_GET_CONST(int64_t, op->GetAttr("keepdims"));
      popart::TensorId result =
          builder_->aiOnnxOpset11().reducemean(inputs, axes, keepdims);
      tensors_.emplace(outputs[0], result);
    } else if (op_type == "Reshape") {
      auto inputs = GetOpInputs(op);
      auto outputs = op->Output("__outputs__");
      popart::TensorId result = builder_->aiOnnxOpset11().reshape(inputs);
      tensors_.emplace(outputs[0], result);
    } else if (op_type == "Softmax") {
      auto inputs = GetOpInputs(op);
      auto outputs = op->Output("__outputs__");
      auto axis = BOOST_GET_CONST(int64_t, op->GetAttr("axis"));
      popart::TensorId result = builder_->aiOnnxOpset11().softmax(inputs, axis);
      tensors_.emplace(outputs[0], result);
    } else if (op_type == "Sum") {
      auto inputs = GetOpInputs(op);
      auto outputs = op->Output("__outputs__");
      popart::TensorId result = builder_->aiOnnxOpset11().sum(inputs);
      tensors_.emplace(outputs[0], result);
    } else if (op_type == "Pow") {
      auto inputs = GetOpInputs(op);
      auto outputs = op->Output("__outputs__");
      popart::TensorId result = builder_->aiOnnxOpset11().pow(inputs);
      tensors_.emplace(outputs[0], result);
    } else if (op_type == "Relu") {
      auto inputs = GetOpInputs(op);
      auto outputs = op->Output("__outputs__");
      auto result = builder_->aiOnnxOpset11().relu(inputs);
      tensors_.emplace(outputs[0], result);
    } else if (op_type == "Tanh") {
      auto inputs = GetOpInputs(op);
      auto outputs = op->Output("__outputs__");
      auto result = builder_->aiOnnxOpset11().tanh(inputs);
      tensors_.emplace(outputs[0], result);
    } else if (op_type == "Gelu") {
      auto inputs = GetOpInputs(op);
      auto outputs = op->Output("__outputs__");
      auto result = builder_->aiGraphcoreOpset1().gelu(inputs);
      tensors_.emplace(outputs[0], result);
    } else if (op_type == "BatchNormalization") {
      auto inputs = GetOpInputs(op);
      auto outputs = op->Output("__outputs__");
      // num_outputs training mode 5, inference mode 1
      auto num_outputs = ipu_strategy_->is_training_ ? 5 : 1;
      auto epsilon = BOOST_GET_CONST(float, op->GetAttr("epsilon"));
      auto momentum = BOOST_GET_CONST(float, op->GetAttr("momentum"));
      auto result = builder_->aiOnnxOpset11().batchnormalization(
          inputs, num_outputs, epsilon, momentum);
      for (int i = 0; i < num_outputs; i++) {
        tensors_.emplace(outputs[i], result[i]);
      }
    } else if (op_type == "GroupNorm") {
      auto inputs = GetOpInputs(op);
      auto outputs = op->Output("__outputs__");
      auto epsilon = BOOST_GET_CONST(float, op->GetAttr("epsilon"));
      auto groups = BOOST_GET_CONST(int64_t, op->GetAttr("groups"));
      std::vector<popart::TensorId> result =
          builder_->aiGraphcoreOpset1().groupnormalization(inputs, groups,
                                                           epsilon);
      // Y
      tensors_.emplace(outputs[0], result[0]);
      // Mean
      tensors_.emplace(outputs[1], result[1]);
      // Variance
      tensors_.emplace(outputs[2], result[2]);
    } else if (op_type == "InstanceNorm") {
      auto inputs = GetOpInputs(op);
      auto outputs = op->Output("__outputs__");
      auto epsilon = BOOST_GET_CONST(float, op->GetAttr("epsilon"));
      popart::TensorId result =
          builder_->aiOnnxOpset11().instancenormalization(inputs, epsilon);
      tensors_.emplace(outputs[0], result);
    } else if (op_type == "MaxPool") {
      auto inputs = GetOpInputs(op);
      auto outputs = op->Output("__outputs__");
      // popart have only one output.
      auto kernel_shape =
          BOOST_GET_CONST(std::vector<int64_t>, op->GetAttr("kernel_shape"));
      auto ceil_mode = BOOST_GET_CONST(bool, op->GetAttr("ceil_mode"));

      auto paddings =
          BOOST_GET_CONST(std::vector<int64_t>, op->GetAttr("paddings"));
      auto storage_order = BOOST_GET_CONST(int, op->GetAttr("storage_order"));
      auto strides =
          BOOST_GET_CONST(std::vector<int64_t>, op->GetAttr("strides"));
      auto result = builder_->aiOnnxOpset11().maxpool(inputs, 1, kernel_shape,
                                                      ceil_mode, {}, paddings,
                                                      storage_order, strides);
      tensors_.emplace(outputs[0], result[0]);
    } else if (op_type == "AveragePool") {
      auto inputs = GetOpInputs(op);
      auto outputs = op->Output("__outputs__");
      // popart have only one output.
      auto kernel_shape =
          BOOST_GET_CONST(std::vector<int64_t>, op->GetAttr("kernel_shape"));
      auto ceil_mode = BOOST_GET_CONST(bool, op->GetAttr("ceil_mode"));

      auto paddings =
          BOOST_GET_CONST(std::vector<int64_t>, op->GetAttr("paddings"));
      auto strides =
          BOOST_GET_CONST(std::vector<int64_t>, op->GetAttr("strides"));
      auto count_include_pad =
          BOOST_GET_CONST(int, op->GetAttr("count_include_pad"));
      auto result = builder_->aiOnnxOpset11().averagepool(
          inputs, kernel_shape, ceil_mode, count_include_pad, paddings,
          strides);

      tensors_.emplace(outputs[0], result);
    } else if (op_type == "Transpose") {
      auto inputs = GetOpInputs(op);
      auto outputs = op->Output("__outputs__");
      auto perm = BOOST_GET_CONST(std::vector<int64_t>, op->GetAttr("axis"));
      auto result = builder_->aiOnnxOpset11().transpose(inputs, perm);

      tensors_.emplace(outputs[0], result);
    } else {
      PADDLE_THROW(platform::errors::Unimplemented("Unimplemented op type %s.",
                                                   op_type));
    }
  }
}

float IpuBackend::GetLRFromScope() {
  auto lr_var = scope_->GetVar(optimizer_.lr_var_name_);
  auto tensor = lr_var->Get<framework::LoDTensor>();

  PADDLE_ENFORCE_EQ(tensor.type(), framework::proto::VarType::FP32,
                    platform::errors::InvalidArgument(
                        "LR requiree float, but got (%s).", tensor.type()));

  return tensor.data<float>()[0];
}

// ipu_num_ must be pow(2,n);
int IpuBackend::UpperIpuNum() {
  PADDLE_ENFORCE_GT(ipu_strategy_->num_ipus_, 0,
                    platform::errors::Unavailable(
                        "The ipu num get is wrong, please make sure the "
                        "sharding or pipline parameter is right."));
  int i = 0;
  while (pow(2, i) < ipu_strategy_->num_ipus_) {
    i++;
  }
  return pow(2, i);
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
  // compile ipu is not same as the runtime ipu.
  VLOG(1) << "comile ipu id = " << id;
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

IpuBackend::~IpuBackend() {
  if (instance_ == nullptr) {
    return;
  }

  // detach device
  if (curr_device_ != nullptr && curr_device_->isAttached()) {
    curr_device_->detach();
  }
}
bool IpuBackend::DeviceIsAttached() { return curr_device_ != nullptr; }

}  // namespace ipu
}  // namespace framework
}  // namespace paddle
