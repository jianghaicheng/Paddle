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

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <popart/sessionoptions.hpp>

#include "boost/optional.hpp"
#include "paddle/fluid/framework/ir/pass_builder.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

namespace ipu {

using VirtualGraphMode = popart::VirtualGraphMode;

struct IpuStrategy {
  int num_ipus_ = 1;
  bool is_training_ = true;
  popart::SessionOptions popart_options_;
};

}  // namespace ipu

}  // namespace framework
}  // namespace paddle
