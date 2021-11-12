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
class TestBase(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_training()
        self.set_feed()
        self.set_feed_attr()
        self.set_attrs()

    def set_feed(self):
        data1 = np.random.uniform(size=[2, 3])
        data2 = np.random.uniform(size=[3, 2])
        self.feed_fp32 = {
            "x": data1.astype('float32'),
            "y": data2.astype('float32'),
        }
        self.feed_fp16 = {
            "x": data1.astype('float16'),
            "y": data2.astype('float16'),
        }

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())
        self.feed_dtype = [
            np_dtype_to_fluid_str(x.dtype) for x in self.feed_fp32.values()
        ]

    def set_attrs(self):
        self.attrs = {
            "transpose_x": False,
            "transpose_y": False,
            "alpha": 1.0,
        }

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
                    out = paddle.fluid.layers.matmul(x, y, **self.attrs)

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

    def test_base(self):
        res0 = self._test_base(self.TEST_CPU_FP32)
        res1 = self._test_base(self.TEST_IPU_FP32)
        res2 = self._test_base(self.TEST_IPU_FP16)

        self.assertTrue(
            np.allclose(
                res0.flatten(), res1.flatten(), atol=self.atol))
        self.assertTrue(
            np.allclose(
                res1.flatten(),
                res2.flatten(),
                atol=self.atol_fp16,
                rtol=self.rtol_fp16))


class TestCase1(TestBase):
    def set_attrs(self):
        self.attrs = {
            "transpose_x": True,
            "transpose_y": True,
            "alpha": 1.0,
        }


class TestCase2(TestBase):
    def set_attrs(self):
        self.attrs = {
            "transpose_x": True,
            "transpose_y": True,
            "alpha": 3.14,
        }

    def set_atol(self):
        self.atol = 1e-10
        self.atol_fp16 = 1e-3
        self.rtol_fp16 = 1e-3


class TestCase3(TestBase):
    def set_feed(self):
        data1 = np.random.uniform(size=[5, 4, 3, 2])
        data2 = np.random.uniform(size=[5, 4, 2, 3])

        self.feed_fp32 = {
            "x": data1.astype('float32'),
            "y": data2.astype('float32'),
        }
        self.feed_fp16 = {
            "x": data1.astype('float16'),
            "y": data2.astype('float16'),
        }


class TestCase4(TestBase):
    def set_feed(self):
        data1 = np.random.uniform(size=[4, 3, 2])
        data2 = np.random.uniform(size=[4, 2, 3])

        self.feed_fp32 = {
            "x": data1.astype('float32'),
            "y": data2.astype('float32'),
        }
        self.feed_fp16 = {
            "x": data1.astype('float16'),
            "y": data2.astype('float16'),
        }


class TestCase5(TestBase):
    def set_feed(self):
        data1 = np.random.uniform(size=[4, 2, 3])
        data2 = np.random.uniform(size=[3, 2])

        self.feed_fp32 = {
            "x": data1.astype('float32'),
            "y": data2.astype('float32'),
        }
        self.feed_fp16 = {
            "x": data1.astype('float16'),
            "y": data2.astype('float16'),
        }


class TestCase6(TestBase):
    def set_feed(self):
        data = np.random.uniform(size=[3])
        self.feed_fp32 = {
            "x": data.astype('float32'),
            "y": data.astype('float32'),
        }
        self.feed_fp16 = {
            "x": data.astype('float16'),
            "y": data.astype('float16'),
        }


class TestCase7(TestBase):
    def set_feed(self):
        data1 = np.random.uniform(size=[1, 12, 128, 64])
        data2 = np.random.uniform(size=[1, 12, 128, 64])

        self.feed_fp32 = {
            "x": data1.astype('float32'),
            "y": data2.astype('float32'),
        }
        self.feed_fp16 = {
            "x": data1.astype('float16'),
            "y": data2.astype('float16'),
        }

    def set_attrs(self):
        self.attrs = {"transpose_x": False, "transpose_y": True, "alpha": 0.125}


@unittest.skip("not supported")
class TestCase6_2(TestCase6):
    def set_feed(self):

        data = np.random.uniform(size=[3])
        self.feed_fp32 = {
            "x": data.astype('float32'),
            "y": data.astype('float32'),
        }
        self.feed_fp16 = {
            "x": data.astype('float16'),
            "y": data.astype('float16'),
        }

    def set_attrs(self):
        self.attrs = {
            "transpose_x": True,
            "transpose_y": True,
            "alpha": 1.0,
        }


class TestCase7(TestBase):
    def set_feed(self):
        data1 = np.random.uniform(size=[3, 1])
        data2 = np.random.uniform(size=[1, 2])

        self.feed_fp32 = {
            "x": data1.astype('float32'),
            "y": data2.astype('float32'),
        }
        self.feed_fp16 = {
            "x": data1.astype('float16'),
            "y": data2.astype('float16'),
        }


@unittest.skip("not supported")
class TestCase7_2(TestBase):
    def set_feed(self):
        data1 = np.random.uniform(size=[3])
        data2 = np.random.uniform(size=[2])

        self.feed_fp32 = {
            "x": data1.astype('float32'),
            "y": data2.astype('float32'),
        }
        self.feed_fp16 = {
            "x": data1.astype('float16'),
            "y": data2.astype('float16'),
        }

    def set_attrs(self):
        self.attrs = {
            "transpose_x": True,
            "transpose_y": True,
            "alpha": 1.0,
        }


@unittest.skip("dim > 4 is not supported")
class TestCase8(TestBase):
    def set_feed(self):
        data = np.random.uniform(size=[6, 5, 4, 2, 3])

        self.feed_fp32 = {
            "x": data.astype('float32'),
            "y": data.astype('float32'),
        }
        self.feed_fp16 = {
            "x": data.astype('float16'),
            "y": data.astype('float16'),
        }


if __name__ == "__main__":
    unittest.main()
