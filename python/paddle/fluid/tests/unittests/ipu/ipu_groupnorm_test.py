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

from __future__ import print_function

import numpy as np
import unittest
import sys
import paddle
import paddle.fluid as fluid
import paddle.fluid.compiler as compiler

paddle.enable_static()
SEED = 2021


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestGroupNormNet(unittest.TestCase):
    def _test(self, run_ipu=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        #paddle only supports 4-D tensor as input
        np_image = np.random.rand(1, 3, 10, 10).astype(np.float32)

        with paddle.static.program_guard(main_prog, startup_prog):
            image = paddle.static.data(
                name='image', shape=[1, 3, 10, 10], dtype='float32')
            conv1 = paddle.static.nn.conv2d(
                image, num_filters=3, filter_size=3, bias_attr=False)

            #paddle groupnorm requires bias_attr and scale_attr
            bias_attr = paddle.ParamAttr(name="bias", trainable=True)
            scale_attr = paddle.ParamAttr(name="scale", trainable=True)

            gn = paddle.static.nn.group_norm(
                input=conv1,
                param_attr=scale_attr,
                bias_attr=bias_attr,
                groups=3)

            loss = paddle.mean(gn)

            adam = paddle.optimizer.Adam(learning_rate=1e-2)
            adam.minimize(loss)

        if run_ipu:
            place = paddle.IPUPlace(0)
        else:
            place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(startup_prog)

        if run_ipu:
            feed_list = [image.name]
            fetch_list = [loss.name]
            ipu_build_strategy = compiler.get_ipu_build_strategy()
            ipu_build_strategy.is_training = True
            program = compiler.IpuCompiler(
                main_prog, ipu_build_strategy=ipu_build_strategy).compile(
                    feed_list, fetch_list)
        else:
            program = main_prog

        result = []
        for epoch in range(100):
            loss_res = exe.run(program,
                               feed={"image": np_image},
                               fetch_list=[loss])
            result.append(loss_res)
        return np.array(result)

    def test_training(self):
        # cpu and ipu dimenstion mismatch, cpu:(100, 1, 1), ipu:(100, 1)
        cpu_loss = self._test(False).flatten()
        ipu_loss = self._test(True).flatten()
        self.assertTrue(np.allclose(ipu_loss, cpu_loss, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
