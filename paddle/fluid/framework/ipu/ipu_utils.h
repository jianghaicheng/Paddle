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
#include <popart/ndarraywrapper.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorinfo.hpp>

#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/lod_tensor.h"

namespace paddle {
namespace framework {

popart::DataType VarType2PopartType(proto::VarType::Type type);

template <typename T>
std::unique_ptr<popart::NDArrayWrapper<T>> Tensor2IArray(Tensor &tensor) {
  auto dtype = VarType2PopartType(tensor.type());
  auto shape = std::vector<int64_t>();
  for (size_t i = 0; i < tensor.dims().size(); ++i) {
    shape.push_back(tensor.dims().at(i));
  }
  popart::TensorInfo tensor_info(dtype, shape);

  return std::make_unique<popart::NDArrayWrapper<T>>(
      reinterpret_cast<T *>(tensor.data<void>()), tensor_info);
}

template <typename T>
std::unique_ptr<popart::NDArrayWrapper<T>> LoDTensor2IArray(
    LoDTensor &lod_tensor) {
  if (lod_tensor.lod().size() == 0) {
    return Tensor2IArray<T>(lod_tensor);
  } else {
    PADDLE_THROW(
        platform::errors::Unimplemented("LoDTensor2IArray is Unimplemented"));
  }
}

}  // namespace framework
}  // namespace paddle