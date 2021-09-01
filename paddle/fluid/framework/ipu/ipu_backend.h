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

#pragma once

#include <popart/adam.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/names.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/optimizer.hpp>
#include <popart/session.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/stepio.hpp>
#include <popart/tensorinfo.hpp>

#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/ipu/compiler.h"
#include "paddle/fluid/framework/ipu/device.h"
#include "paddle/fluid/framework/ipu/ipu_strategy.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ipu {

using ipu::IpuStrategy;
struct Optimizer {
  std::string type_;
  std::string loss_;
  std::string lr_var_name_;
  // as far as we know, attr is usually float
  std::map<std::string, float> attrs_;
};

class IpuBackend {
  // IpuBackend is the center of paddle-ipu, its function include:
  //   1. Compile paddle model to popart model
  //   2. Run popart model, inference or training
  //   3. Request and release device
  //   4. Other helper function

 public:
  IpuBackend();
  ~IpuBackend();

  static std::shared_ptr<IpuBackend> GetInstance();

  void Compile(ir::Graph *graph, const std::vector<std::string> &feed_list,
               const std::vector<std::string> &fetch_list);
  void Run(const std::vector<const Tensor *> &inputs,
           const std::vector<Tensor *> &outputs);

  std::vector<int64_t> GetTensorShape(const std::string &var_name);
  void SetScope(const Scope &scope) { scope_ = &scope; }

  // Optimizer
  std::unique_ptr<popart::Optimizer> GetPopartOptimizer();
  std::string GetOptimizerType() { return optimizer_.type_; }
  void SetOptimizerType(const std::string &type) { optimizer_.type_ = type; }
  float GetOptimizerAttr(const std::string &attr, float default_value = 0.0f);
  void SetOptimizerAttr(const std::string &attr, float value);
  void SetLoss(const std::string &loss) { optimizer_.loss_ = loss; }
  void SetLRVarName(const std::string &name) { optimizer_.lr_var_name_ = name; }

  // IpuStrategy
  void SetIpuStrategy(const IpuStrategy &strategy);
  size_t GetNumDevices();
  std::vector<int> GetDeviceIds();
  Device GetDevice(int id);
  void AttachDevice(int id);
  bool DeviceIsAttached();

 private:
  void Prepare();
  float GetLRFromScope();
  int UpperIpuNum();

 private:
  static std::shared_ptr<IpuBackend> instance_;
  std::shared_ptr<Compiler> compiler_;

  Optimizer optimizer_;
  std::unique_ptr<popart::Session> session_;
  std::shared_ptr<popart::DeviceInfo> curr_device_;
  bool is_prepared_ = false;

  // not own
  const Scope *scope_ = nullptr;
  const IpuStrategy *ipu_strategy_ = nullptr;
};

}  // namespace ipu
}  // namespace framework
}  // namespace paddle
