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
class TestGatherNet(unittest.TestCase):
    def _test(self, run_ipu=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        np_image = np.random.rand(3, 2).astype(np.float32)
        np_index = np.array([1, 2]).astype(np.int32)

        with paddle.static.program_guard(main_prog, startup_prog):
            image = paddle.static.data(
                name='image', shape=[3, 2], dtype='float32')
            index = paddle.static.data(name='index', shape=[2], dtype='int32')
            gather = paddle.fluid.layers.gather(image, index)

        if run_ipu:
            place = paddle.IPUPlace()
        else:
            place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(startup_prog)

        if run_ipu:
            feed_list = [image.name, index.name]
            fetch_list = [gather.name]
            ipu_strategy = compiler.get_ipu_strategy()
            ipu_strategy.is_training = False
            program = compiler.IpuCompiler(
                main_prog, ipu_strategy=ipu_strategy).compile(feed_list,
                                                              fetch_list)
        else:
            program = main_prog

        result = exe.run(program,
                         feed={"image": np_image,
                               "index": np_index},
                         fetch_list=[gather])
        return result[0]

    def test_gather(self):
        cpu = self._test(False)
        print(cpu.shape)
        ipu = self._test(True)
        print(ipu.shape)
        self.assertTrue(np.allclose(ipu, cpu, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
