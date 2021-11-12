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
class TestMean(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_training()
        self.init_op()

    def init_op(self):
        self.op = paddle.fluid.layers.reduce_mean

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed.values()]
        self.feed_list = list(self.feed.keys())
        self.feed_dtype = [
            np_dtype_to_fluid_str(x.dtype) for x in self.feed.values()
        ]

    def _test_base(self, run_mode=0):
        scope = fluid.core.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        SEED = self.SEED
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        dtype = 'float32' if run_mode != self.TEST_IPU_FP16 else 'float16'

        self.feed_fp16 = {"in_0": self.feed["in_0"].astype(np.float16)}
        feed = self.feed_fp16 if run_mode == self.TEST_IPU_FP16 else self.feed

        with fluid.scope_guard(scope):
            with paddle.static.program_guard(main_prog, startup_prog):
                with paddle.static.amp.fp16_guard():
                    x = paddle.static.data(
                        name=self.feed_list[0],
                        shape=self.feed_shape[0],
                        dtype=dtype)
                    out = self.op(x, **self.attrs)

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
        res2 = self._test_base(self.TEST_IPU_FP16)

        self.assertTrue(
            np.allclose(
                res0.flatten(), res1.flatten(), atol=self.atol))
        self.assertTrue(
            np.allclose(
                res1.flatten(), res2.flatten(), atol=self.atol_fp16))

        self.assertTrue(res0.shape == res1.shape)

    def set_feed0(self):
        self.feed = {}
        self.feed["in_0"] = np.random.uniform(size=[2, 4]).astype(np.float32)
        self.set_feed_attr()

    def set_feed1(self):
        self.feed = {}
        self.feed["in_0"] = np.random.uniform(size=[2, 2, 2]).astype(np.float32)
        self.set_feed_attr()

    def set_attr0(self):
        self.attrs = {}
        self.attrs['dim'] = None
        self.attrs['keep_dim'] = False

    def test_case0(self):
        self.set_feed0()
        self.set_attr0()
        self.run_test_base()

    def test_case1(self):
        self.set_feed0()
        self.set_attr0()
        self.attrs['dim'] = 0
        self.run_test_base()

    def test_case2(self):
        self.set_feed0()
        self.set_attr0()
        self.attrs['dim'] = -1
        self.run_test_base()

    def test_case3(self):
        self.set_feed0()
        self.set_attr0()
        self.attrs['dim'] = 1
        self.run_test_base()

    def test_case4(self):
        self.set_feed0()
        self.attrs = {}
        self.attrs['dim'] = 1
        self.attrs['keep_dim'] = True
        self.run_test_base()

    def test_case5(self):
        self.set_feed1()
        self.attrs = {}
        self.attrs['dim'] = [1, 2]
        self.attrs['keep_dim'] = False
        self.run_test_base()

    def test_case6(self):
        self.set_feed1()
        self.attrs = {}
        self.attrs['dim'] = [0, 1]
        self.attrs['keep_dim'] = False
        self.run_test_base()

    def test_case7(self):
        self.set_feed1()
        self.attrs = {}
        self.attrs['dim'] = [0, 1]
        self.attrs['keep_dim'] = True
        self.run_test_base()


class TestMax(TestMean):
    def init_op(self):
        self.op = paddle.fluid.layers.reduce_max


class TestMin(TestMean):
    def init_op(self):
        self.op = paddle.fluid.layers.reduce_min


class TestProd(TestMean):
    def init_op(self):
        self.op = paddle.fluid.layers.reduce_prod


class TestSum(TestMean):
    def init_op(self):
        self.op = paddle.fluid.layers.reduce_sum


if __name__ == "__main__":
    unittest.main()
