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

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.compiler as compiler
import paddle.optimizer
import paddle.static
from paddle.fluid.tests.unittests.ipu.op_test_ipu import (IPUOpTest,
                                                          np_dtype_to_fluid_str)
import paddle.fluid.contrib.mixed_precision.fp16_utils as fp16_utils

paddle.enable_static()


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestMul(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_training()
        self.init_op()

    def init_op(self):
        self.op = paddle.fluid.layers.elementwise_mul

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def _test_base(self, run_mode=0):
        scope = fluid.core.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        SEED = self.SEED
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        dtype = 'float32' if run_mode != self.TEST_IPU_FP16 else 'float16'
        feed = self.feed_fp16 if run_mode == self.TEST_IPU_FP16 else self.feed_fp32

        with fluid.scope_guard(scope):
            with paddle.static.program_guard(main_prog, startup_prog):
                with paddle.static.amp.fp16_guard():
                    x = paddle.static.data(
                        name=self.feed_list[0],
                        shape=self.feed_shape[0],
                        dtype=dtype)
                    y = paddle.static.data(
                        name=self.feed_list[1],
                        shape=self.feed_shape[1],
                        dtype=dtype)
                    out = self.op(x, y, **self.attrs)

                fetch_list = [out.name]

            if run_mode != self.TEST_CPU_FP32:
                place = paddle.IPUPlace()
            else:
                place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            if run_mode == self.TEST_IPU_FP16:
                fp16_utils.rewrite_program_v2(
                    startup_prog=startup_prog,
                    main_prog=main_prog,
                    amp_lists=self.amp_list)
            exe.run(startup_prog)

            if run_mode != self.TEST_CPU_FP32:
                feed_list = self.feed_list
                ipu_strategy = compiler.get_ipu_strategy()
                ipu_strategy.is_training = self.is_training
                if self.op == paddle.fluid.layers.elementwise_div:
                    ipu_strategy.save_init_onnx = True
                program = compiler.IpuCompiler(
                    main_prog,
                    ipu_strategy=ipu_strategy).compile(feed_list, fetch_list)
            else:
                program = main_prog

            result = exe.run(program, feed=feed, fetch_list=fetch_list)
            return result[0]

    def run_test_base(self):
        res0 = self._test_base(self.TEST_CPU_FP32)
        res1 = self._test_base(self.TEST_IPU_FP32)

        self.assertTrue(
            np.allclose(
                res0.flatten(), res1.flatten(), atol=self.atol))

        res2 = self._test_base(self.TEST_IPU_FP16)
        # TestDiv and Test Mod decreased the accuracy!!!
        self.assertTrue(
            np.allclose(
                res1.flatten(),
                res2.flatten(),
                atol=self.atol_fp16,
                rtol=self.rtol_fp16))

    def test_case0(self):
        data_x = np.random.uniform(size=(2, 3, 4, 5))
        data_y = np.random.uniform(size=(2, 3, 4, 5))

        self.feed_fp32 = {
            "x": data_x.astype('float32'),
            "y": data_y.astype('float32'),
        }
        self.feed_fp16 = {
            "x": data_x.astype('float16'),
            "y": data_y.astype('float16'),
        }
        self.attrs = {}
        self.set_feed_attr()
        self.run_test_base()

    def test_case1(self):
        data_x = np.random.uniform(size=(2, 3, 4, 5))
        data_y = np.random.uniform(size=(3, 4))
        self.feed_fp32 = {
            "x": data_x.astype('float32'),
            "y": data_y.astype('float32'),
        }
        self.feed_fp16 = {
            "x": data_x.astype('float16'),
            "y": data_y.astype('float16'),
        }
        self.set_feed_attr()
        self.attrs = {"axis": 1}
        self.run_test_base()

    def test_case2(self):
        data_x = np.random.uniform(size=(2, 3, 4, 5))
        data_y = np.random.uniform(size=(5))
        self.feed_fp32 = {
            "x": data_x.astype('float32'),
            "y": data_y.astype('float32'),
        }
        self.feed_fp16 = {
            "x": data_x.astype('float16'),
            "y": data_y.astype('float16'),
        }
        self.set_feed_attr()
        self.attrs = {"axis": -1}
        self.run_test_base()

    def test_case3(self):
        data_x = np.random.uniform(size=(2, 3, 4, 5))
        data_y = np.random.uniform(size=(2))
        self.feed_fp32 = {
            "x": data_x.astype('float32'),
            "y": data_y.astype('float32'),
        }
        self.feed_fp16 = {
            "x": data_x.astype('float16'),
            "y": data_y.astype('float16'),
        }
        self.set_feed_attr()
        self.attrs = {"axis": 0}
        self.run_test_base()


class TestAdd(TestMul):
    def init_op(self):
        self.op = paddle.fluid.layers.elementwise_add


class TestSub(TestMul):
    def init_op(self):
        self.op = paddle.fluid.layers.elementwise_sub


class TestDiv(TestMul):
    def init_op(self):
        self.op = paddle.fluid.layers.elementwise_div


class TestMin(TestMul):
    def init_op(self):
        self.op = paddle.fluid.layers.elementwise_min


class TestMax(TestMul):
    def init_op(self):
        self.op = paddle.fluid.layers.elementwise_max


class TestPow(TestMul):
    def init_op(self):
        self.op = paddle.fluid.layers.elementwise_pow


class TestMod(TestMul):
    def init_op(self):
        self.op = paddle.fluid.layers.elementwise_mod

    def set_atol(self):
        self.atol = 1e-7
        self.atol_fp16 = 1e-3
        self.rtol_fp16 = 1e-3


if __name__ == "__main__":
    unittest.main()
