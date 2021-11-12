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
        self.set_attrs()

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed.values()]
        self.feed_list = list(self.feed.keys())
        self.feed_dtype = [
            np_dtype_to_fluid_str(x.dtype) for x in self.feed.values()
        ]

    def set_attrs(self):
        self.attrs = {
            "size": [128, 16],
            "is_sparse": False,
            "is_distributed": False,
            "padding_idx": -1,
            "dtype": 'float32'
        }

    def _test_base(self, run_mode=0):
        scope = fluid.core.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        SEED = self.SEED
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        if run_mode != self.TEST_CPU_FP32:
            self.feed = {
                "x": np.array(
                    [[[1], [3]], [[2], [4]], [[4], [127]]]).astype(np.int32)
            }
        else:
            self.feed = {
                "x": np.array(
                    [[[1], [3]], [[2], [4]], [[4], [127]]]).astype(np.int64)
            }

        self.set_feed_attr()

        with fluid.scope_guard(scope):
            with paddle.static.program_guard(main_prog, startup_prog):
                with paddle.static.amp.fp16_guard():
                    x = paddle.static.data(
                        name=self.feed_list[0],
                        shape=self.feed_shape[0],
                        dtype=self.feed_dtype[0])
                    out = paddle.fluid.layers.embedding(x, **self.attrs)

                    if self.is_training:
                        loss = paddle.mean(out)
                        adam = paddle.optimizer.Adam(learning_rate=1e-2)
                        adam.minimize(loss)
                        fetch_list = [loss.name]
                    else:
                        fetch_list = [out.name]

            if run_mode != self.TEST_CPU_FP32:
                place = paddle.IPUPlace()
            else:
                place = paddle.CPUPlace()
            if run_mode == self.TEST_IPU_FP16 and self.is_training == False:
                fp16_utils.rewrite_program_v2(
                    startup_prog=startup_prog,
                    main_prog=main_prog,
                    amp_lists=self.amp_list)
            exe = paddle.static.Executor(place)
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

            if self.is_training:
                result = []
                for _ in range(self.epoch):
                    loss_res = exe.run(program,
                                       feed=self.feed,
                                       fetch_list=fetch_list)
                    result.append(loss_res[0])
                return np.array(result)
            else:
                result = exe.run(program, feed=self.feed, fetch_list=fetch_list)
                return result[0]

    def test_base(self):
        res0 = self._test_base(self.TEST_CPU_FP32)
        res1 = self._test_base(self.TEST_IPU_FP32)
        res2 = self._test_base(self.TEST_IPU_FP16)

        self.assertTrue(
            np.allclose(
                res0.flatten(), res1.flatten(), atol=self.atol))
        self.assertTrue(res0.shape == res1.shape)
        self.assertTrue(
            np.allclose(
                res1.flatten(),
                res2.flatten(),
                rtol=self.rtol_fp16,
                atol=self.atol_fp16))


class TestTrainCase1(TestBase):
    def set_atol(self):
        self.atol = 1e-7
        self.atol_fp16 = 1e-3
        self.rtol_fp16 = 1e-3

    def set_training(self):
        self.is_training = True
        self.epoch = 10


if __name__ == "__main__":
    unittest.main()
