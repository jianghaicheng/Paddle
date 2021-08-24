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

#define ReduceMeanHandler                                                     \
  [&](OpDesc *op_desc) {                                                      \
    auto inputs = GetOpInputs(op_desc);                                       \
    auto outputs = op_desc->Output("__outputs__");                            \
    auto axes = nonstd::optional<std::vector<int64_t>>();                     \
    if (op_desc->HasAttr("axes")) {                                           \
      axes = BOOST_GET_CONST(std::vector<int64_t>, op_desc->GetAttr("axes")); \
    }                                                                         \
    auto keepdims = BOOST_GET_CONST(int64_t, op_desc->GetAttr("keepdims"));   \
    popart::TensorId result =                                                 \
        builder_->aiOnnxOpset11().reducemean(inputs, axes, keepdims);         \
    tensors_.emplace(outputs[0], result);                                     \
  }

#define BatchNormHandler                                                  \
  [&](OpDesc *op_desc) {                                                  \
    auto inputs = GetOpInputs(op_desc);                                   \
    auto outputs = op_desc->Output("__outputs__");                        \
    /*num_outputs training mode 5, inference mode 1*/                     \
    auto num_outputs = ipu_strategy_->is_training_ ? 5 : 1;               \
    auto epsilon = BOOST_GET_CONST(float, op_desc->GetAttr("epsilon"));   \
    auto momentum = BOOST_GET_CONST(float, op_desc->GetAttr("momentum")); \
    auto result = builder_->aiOnnxOpset11().batchnormalization(           \
        inputs, num_outputs, epsilon, momentum);                          \
    for (int i = 0; i < num_outputs; i++) {                               \
      tensors_.emplace(outputs[i], result[i]);                            \
    }                                                                     \
  }

#define Constant                                                               \
  [&](OpDesc *op_desc) {                                                       \
    auto outputs = op_desc->Output("__outputs__");                             \
    auto dims =                                                                \
        BOOST_GET_CONST(std::vector<int64_t>, op_desc->GetAttr("dims"));       \
    auto dtype_ = BOOST_GET_CONST(int, op_desc->GetAttr("dtype"));             \
    auto dtype = OnnxDtype2PopartType(dtype_);                                 \
    popart::TensorInfo tensor_info{dtype, dims};                               \
    auto value_attr = op_desc->GetAttr("value");                               \
    auto const_data = std::unique_ptr<popart::ConstVoidData>{};                \
    switch (dtype) {                                                           \
      case popart::DataType::FLOAT:                                            \
        const_data.reset(new popart::ConstVoidData(                            \
            BOOST_GET_CONST(std::vector<float>, value_attr).data(),            \
            tensor_info));                                                     \
        break;                                                                 \
      case popart::DataType::INT32:                                            \
        const_data.reset(new popart::ConstVoidData(                            \
            BOOST_GET_CONST(std::vector<int>, value_attr).data(),              \
            tensor_info));                                                     \
        break;                                                                 \
      case popart::DataType::DOUBLE:                                           \
        const_data.reset(new popart::ConstVoidData(                            \
            BOOST_GET_CONST(std::vector<double>, value_attr).data(),           \
            tensor_info));                                                     \
        break;                                                                 \
      case popart::DataType::INT64:                                            \
        const_data.reset(new popart::ConstVoidData(                            \
            BOOST_GET_CONST(std::vector<int64_t>, value_attr).data(),          \
            tensor_info));                                                     \
        break;                                                                 \
      default:                                                                 \
        PADDLE_THROW(                                                          \
            platform::errors::Unimplemented("popart::DataType %d", dtype));    \
    }                                                                          \
    popart::TensorId result = builder_->aiOnnxOpset11().constant(*const_data); \
    tensors_.emplace(outputs[0], result);                                      \
  }
