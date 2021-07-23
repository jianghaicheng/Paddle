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

#include "paddle/fluid/framework/ipu/ipu_utils.h"

namespace paddle {
namespace framework {

popart::DataType VarType2PopartType(proto::VarType::Type type) {
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
      PADDLE_THROW(paddle::platform::errors::Unavailable(
          "Unsupported Paddle var type."));
  }
}

}  // namespace framework
}  // namespace paddle