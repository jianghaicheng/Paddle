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
import paddle.nn.functional as F
import paddle.static.amp as amp
import paddle.fluid.contrib.mixed_precision.fp16_utils as fp16_utils
from paddle.fluid.tests.unittests.ipu.op_test_ipu import IPUOpTest
paddle.enable_static()
import datetime


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestBase(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_feed()

    def set_feed(self):
        self.feed_shape = []
        self.feed_shape.append([1, 10, 27, 27])

        self.feed_cpu = {}
        self.feed_ipu = {}
        feed = np.random.uniform(size=self.feed_shape[0])
        self.feed_cpu["in_0"] = feed.astype(np.float32)
        self.feed_ipu["in_0"] = feed.astype(np.float16)
        self.feed_list = list(self.feed_cpu.keys())

    def _test_base(self, run_ipu=True):
        generator = fluid.unique_name.UniqueNameGenerator()
        scope = fluid.core.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()

        SEED = self.SEED
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED

        with fluid.unique_name.guard(generator):
            with fluid.scope_guard(scope):
                with paddle.static.program_guard(main_prog, startup_prog):
                    with paddle.static.amp.fp16_guard():
                        data = paddle.static.data(
                            name=self.feed_list[0],
                            shape=self.feed_shape[0],
                            dtype='float16' if run_ipu else 'float32')
                        conv2d = paddle.static.nn.conv2d(
                            input=data, num_filters=3, filter_size=3)
                    pool_0 = F.max_pool2d(conv2d, kernel_size=2, stride=2)
                    with paddle.static.amp.fp16_guard():
                        conv2d1 = paddle.static.nn.conv2d(
                            input=pool_0, num_filters=6, filter_size=3)
                        bn = paddle.static.nn.batch_norm(
                            input=conv2d1, act="relu")
                    pool = F.max_pool2d(bn, kernel_size=2, stride=2)
                    with paddle.static.amp.fp16_guard():
                        hidden = paddle.static.nn.fc(pool, size=10)
                    loss = paddle.mean(hidden)
                    # ipu rewrite program
                    if run_ipu:
                        # define which op should run with float32
                        amp_list = amp.CustomOpLists(custom_black_list=[
                            # 'conv2d', 'batch_norm', 'reduce_mean',
                        ])
                        fp16_utils.rewrite_program_v2(
                            startup_prog=startup_prog,
                            main_prog=main_prog,
                            amp_lists=amp_list)
                    place = paddle.IPUPlace()
                    exe = paddle.static.Executor(place)
                    exe.run(startup_prog)
                    # strategy 
                    ipu_strategy = compiler.get_ipu_strategy()
                    ipu_strategy.is_training = False
                    # ipu compile
                    fetch_list = [loss.name]
                    if run_ipu:
                        program = compiler.IpuCompiler(
                            main_prog, ipu_strategy=ipu_strategy).compile(
                                self.feed_list, fetch_list)
                    # return result
                    return exe.run(program if run_ipu else main_prog,
                                   feed=self.feed_ipu
                                   if run_ipu else self.feed_cpu,
                                   fetch_list=fetch_list)

    def test_base(self):
        result_ipu = self._test_base(True)
        result_cpu = self._test_base(False)
        mae = np.mean(
            np.abs(
                np.asarray(result_cpu).flatten() - np.asarray(result_ipu)
                .flatten()))
        print('result_cpu:{} result_ipu:{} mae:{} '.format(
            np.asarray(result_cpu).flatten(),
            np.asarray(result_ipu).flatten(), mae))


if __name__ == "__main__":
    unittest.main()
