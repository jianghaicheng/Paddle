/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <boost/preprocessor/repetition/repeat.hpp>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/eigen/eigen_function.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/device_context.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;

using Array1 = Eigen::DSizes<Eigen::DenseIndex, 1>;
using Array2 = Eigen::DSizes<Eigen::DenseIndex, 2>;

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class GCTestKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<Tensor>("Input");
    const Tensor* x = context.Input<Tensor>("X");
    const Tensor* y = context.Input<Tensor>("Y");

    auto input_dims = input->dims();
    auto x_dims = x->dims();
    auto y_dims = y->dims();

    // broadcast mode check
    if (x_dims[0] != input_dims[0]) {
      PADDLE_ENFORCE_EQ(input_dims[0], 1,
                        platform::errors::InvalidArgument(
                            "When x_dims[0] is not equal with input_dims[0], "
                            "input_dims[0] must be 1 but got %s",
                            input_dims[0]));
      PADDLE_ENFORCE_EQ(
          y_dims[1] == input_dims[1] || input_dims[1] == 1, true,
          platform::errors::InvalidArgument(
              "The input tensor shape mismatch, input shape=[%s], "
              "x shape=[%s], y shape=[%s]",
              input_dims, x_dims, y_dims));
    }
    // broadcast mode check

    // blas.GEMM(false, false, x_dims[0], y_dims[1], x_dims[1], alpha,
    //           x->data<T>(), x_dims[1], y->data<T>(), y_dims[1], beta,
    //           out->data<T>(), y_dims[1]);

    VLOG(1) << "success come to compute gctest!!";
  }
};

template <typename DeviceContext, typename T>
class GCTestGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* dout = ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));
    auto in_dims = ctx.Input<framework::LoDTensor>("Input")->dims();
    auto* dinput =
        ctx.Output<framework::LoDTensor>(framework::GradVarName("Input"));
    auto* dx = ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<framework::LoDTensor>(framework::GradVarName("Y"));

    float alpha = ctx.Attr<float>("Alpha");
    float beta = ctx.Attr<float>("Beta");

    int total_elems = 0;

    VLOG(3) << "alpha: " << alpha << " beta: " << beta;

    if (dinput != nullptr) {
      dinput->set_lod(dout->lod());
    }
    if (dx != nullptr) {
      dx->set_lod(x->lod());
    }
    if (dy != nullptr) {
      dy->set_lod(y->lod());
    }

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    if (platform::is_ipu_place(ctx.GetPlace())) {
      VLOG(10) << "1. Succeed on IPU PlatForm.";
    }
    if (platform::is_ipu_place(dev_ctx.GetPlace())) {
      VLOG(10) << "2. Succeed on IPU PlatForm.";
    }

    // auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);
    if (dinput) {
      dinput->mutable_data<T>(ctx.GetPlace());
      total_elems = in_dims[0] * in_dims[1];
      auto& place =
          *ctx.template device_context<DeviceContext>().eigen_device();
      auto eigen_dout = EigenTensor<T, 2>::From(*dout);
      auto eigen_dinput = EigenTensor<T, 2>::From(*dinput);

      bool row_compress = in_dims[0] != dout->dims()[0];
      bool col_compress = in_dims[1] != dout->dims()[1];
      auto eigen_dinput_shape = Array2(dinput->dims()[0], dinput->dims()[1]);

      if (row_compress && col_compress) {
        eigen_dinput.device(place) =
            eigen_dout.sum().eval().reshape(eigen_dinput_shape);
      } else if (row_compress) {
        eigen_dinput.device(place) =
            eigen_dout.sum(Array1(0)).eval().reshape(eigen_dinput_shape);
      } else if (col_compress) {
        eigen_dinput.device(place) =
            eigen_dout.sum(Array1(1)).eval().reshape(eigen_dinput_shape);
      } else {
        // blas.VCOPY(total_elems, dout->data<T>(), dinput->data<T>());
      }

      // blas.SCAL(total_elems, beta, dinput->data<T>());
    }
    if (dx) {
      dx->mutable_data<T>(ctx.GetPlace());
      total_elems = x->dims()[0] * x->dims()[1];
      // dx = dout * y'. dx: M x K, dout : M x N, y : K x N
      // blas.MatMul(*dout, false, *y, true, dx);
      // blas.SCAL(total_elems, alpha, dx->data<T>());
      VLOG(10) << "total_elems" << total_elems;
    }
    if (dy) {
      dy->mutable_data<T>(ctx.GetPlace());
      total_elems = x->dims()[1] * y->dims()[1];
      // dy = x' * dout. dy K x N, dout : M x N, x : M x K
      // blas.MatMul(*x, true, *dout, false, dy);
      // blas.SCAL(total_elems, alpha, dy->data<T>());
      VLOG(10) << "total_elems" << total_elems;
    }
  }
};

}  // namespace operators
}  // namespace paddle
