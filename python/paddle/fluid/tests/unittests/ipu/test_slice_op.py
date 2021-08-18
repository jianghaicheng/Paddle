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
import paddle
import paddle.fluid
import paddle.fluid as fluid
import paddle.fluid.compiler as compiler

paddle.enable_static()
SEED = 2021


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestSlice(unittest.TestCase):
    def _test_slice(self, run_ipu=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        np_data = np.random.uniform(
            low=0, high=1, size=(3, 4, 5, 6)).astype(np.float32)
        with paddle.static.program_guard(main_prog, startup_prog):
            data = fluid.layers.data(
                name="x",
                shape=[3, 4, 5, 6],
                append_batch_size=False,
                dtype="float32")
            out = fluid.layers.slice(
                data, axes=[0, 1, 2], starts=[0, 0, 2], ends=[23333, 100, -1])

        if run_ipu:
            place = paddle.IPUPlace()
        else:
            place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(startup_prog)

        if run_ipu:
            feed_list = [data.name]
            fetch_list = [out.name]
            ipu_strategy = compiler.get_ipu_strategy()
            ipu_strategy.is_training = False
            program = compiler.IpuCompiler(
                main_prog, ipu_strategy=ipu_strategy).compile(feed_list,
                                                              fetch_list)
        else:
            program = main_prog

        result = exe.run(program, feed={data.name: np_data}, fetch_list=[out])
        return result[0]

    def test_slice(self):
        cpu_res = self._test_slice(False)
        ipu_res = self._test_slice(True)

        self.assertTrue(np.allclose(ipu_res, cpu_res, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
