#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import os
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.compiler as compiler
import paddle.optimizer
import paddle.static
from paddle.utils.cpp_extension import load
from paddle.fluid.tests.unittests.ipu.op_test_ipu import (IPUOpTest,
                                                          np_dtype_to_fluid_str)

paddle.enable_static()

enable_test = False

if enable_test:
    # load custom ops
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    custom_ops = load(
        name="custom_jit_ops",
        sources=[
            f"{cur_dir}/leaky_relu_cpu.cc",
            f"{cur_dir}/leaky_relu_ipu.cc",
        ],
        extra_cxx_cflags=['-DONNX_NAMESPACE=onnx'])

    # build a list of IpuCustomOpIdentifier
    # `custom_leaky_relu` was defined in leaky_relu_cpu.cc
    # `LeakyRelu` was defined in leaky_relu_ipu.cc
    custom_ops_list = [
        compiler.IpuCustomOpIdentifier("custom_leaky_relu", "LeakyRelu"),
    ]


@unittest.skipIf(not enable_test, "disable in CI")
class TestBase(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_training()
        self.set_feed()
        self.set_feed_attr()
        self.set_attrs()

    def set_feed(self):
        self.feed = {
            "x": np.random.uniform(
                low=-2, high=2, size=[3, 5]).astype('float32'),
        }

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed.values()]
        self.feed_list = list(self.feed.keys())
        self.feed_dtype = [
            np_dtype_to_fluid_str(x.dtype) for x in self.feed.values()
        ]

    def set_attrs(self):
        self.attrs = {'alpha': 0.1}

    def _test_base(self, run_ipu=True):
        scope = fluid.core.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        SEED = self.SEED
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED

        with fluid.scope_guard(scope):
            with paddle.static.program_guard(main_prog, startup_prog):
                x = paddle.static.data(
                    name=self.feed_list[0],
                    shape=self.feed_shape[0],
                    dtype=self.feed_dtype[0])
                # custom op
                out = custom_ops.custom_leaky_relu(x, **self.attrs)
                fetch_list = [out.name]

            if run_ipu:
                place = paddle.IPUPlace()
            else:
                place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(startup_prog)

            if run_ipu:
                feed_list = self.feed_list
                ipu_strategy = compiler.get_ipu_strategy()
                ipu_strategy.is_training = self.is_training
                program = compiler.IpuCompiler(
                    main_prog,
                    ipu_strategy=ipu_strategy,
                    custom_ops=custom_ops_list).compile(feed_list, fetch_list)
            else:
                program = main_prog

            result = exe.run(program, feed=self.feed, fetch_list=fetch_list)
            return result[0]

    def test_base(self):
        res0 = self._test_base(False)
        res1 = self._test_base(True)

        self.assertTrue(
            np.allclose(
                res0.flatten(), res1.flatten(), atol=self.atol))

        self.assertTrue(res0.shape == res1.shape)


if __name__ == "__main__":
    unittest.main()
