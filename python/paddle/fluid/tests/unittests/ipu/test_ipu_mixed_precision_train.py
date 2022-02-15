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
import paddle.static
import paddle.nn.functional as F
from paddle.fluid.tests.unittests.ipu.op_test_ipu import IPUOpTest, ExecutionMode


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestBase(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_training()
        self.set_data_feed()
        self.set_feed_attr()

    @property
    def fp16_enabled(self):
        return True

    def set_atol(self):
        self.atol = 2e-6
        self.rtol = 1e-5
        self.atol_fp16 = 1e-2
        self.rtol_fp16 = 1e-3

    def set_training(self):
        self.is_training = True
        self.epoch = 20

    def set_data_feed(self):
        data = np.random.uniform(size=[1, 3, 28, 28])
        self.feed_fp32 = {"in_0": data.astype(np.float32)}
        self.feed_fp16 = {"in_0": data.astype(np.float16)}

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def dtype_check(self, program):
        block = program.global_block()
        assert (block.var("conv2d_0.w_0").dtype, paddle.float16)
        assert (block.var("conv2d_0.w_0@GRAD").dtype, paddle.float16)
        assert (block.var("conv2d_0.w_0_moment1_0").dtype, paddle.float32)
        assert (block.var("conv2d_0.w_0_beta2_pow_acc_0").dtype, paddle.float32)

    def _test_base(self, exec_mode):
        generator = paddle.fluid.unique_name.UniqueNameGenerator()
        scope = paddle.fluid.core.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = self.SEED
        startup_prog.random_seed = self.SEED

        with paddle.fluid.unique_name.guard(generator):
            with paddle.fluid.scope_guard(scope):
                with paddle.static.program_guard(main_prog, startup_prog):
                    data = paddle.static.data(
                        name=self.feed_list[0],
                        shape=self.feed_shape[0],
                        dtype='float32')

                    with paddle.static.amp.fp16_guard():
                        conv2d = paddle.static.nn.conv2d(
                            input=data, num_filters=6, filter_size=3)
                        conv2d1 = paddle.static.nn.conv2d(
                            input=conv2d, num_filters=6, filter_size=3)

                        pool_0 = F.max_pool2d(conv2d1, kernel_size=2, stride=2)
                        bn = paddle.static.nn.batch_norm(
                            input=pool_0, act="relu")
                        pool = F.max_pool2d(bn, kernel_size=2, stride=2)
                        loss = paddle.mean(pool)
                        optimizer = paddle.optimizer.Adam(learning_rate=1e-2)

                    if exec_mode == ExecutionMode.IPU_PADDLE_FP16:
                        amp_list = paddle.static.amp.CustomOpLists(
                            custom_black_list=[
                                # 'conv2d', 'batch_norm', 'reduce_mean'
                            ])
                        optimizer = paddle.static.amp.decorate(
                            optimizer,
                            amp_list,
                            init_loss_scaling=1.0,
                            use_dynamic_loss_scaling=False,
                            use_pure_fp16=False)
                        optimizer.minimize(loss, startup_prog)
                        self.dtype_check(main_prog)
                    else:
                        optimizer.minimize(loss, startup_prog)

                fetch_list = [loss.name]

                if exec_mode == ExecutionMode.CPU_FP32:
                    place = paddle.CPUPlace()
                else:
                    place = paddle.IPUPlace()

                exe = paddle.static.Executor(place)
                exe.run(startup_prog)

                if exec_mode != ExecutionMode.CPU_FP32:
                    ipu_strategy = paddle.static.IpuStrategy()
                    ipu_strategy.SetGraphConfig(is_training=self.is_training)
                    if exec_mode == ExecutionMode.IPU_POPART_FP16:
                        ipu_strategy.SetHalfConfig(enable_fp16=True)
                    program = paddle.static.IpuCompiledProgram(
                        main_prog, ipu_strategy=ipu_strategy).compile(
                            self.feed_list, fetch_list)
                else:
                    program = main_prog

                feed = self.feed_fp32
                if exec_mode > ExecutionMode.IPU_FP32:
                    feed = self.feed_fp16

                result = []
                for i in range(self.epoch):
                    out = exe.run(program, feed=feed, fetch_list=fetch_list)
                    result.append(out)
                return np.array(result)

    def test_base(self):
        output_dict = {}
        for mode in ExecutionMode:
            if mode > ExecutionMode.IPU_FP32 and not self.fp16_enabled:
                break
            output_dict[mode] = self._test_base(mode).flatten()

        self.check(output_dict)


if __name__ == "__main__":
    unittest.main()
