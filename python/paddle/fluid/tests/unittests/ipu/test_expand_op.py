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
class TestExpand(unittest.TestCase):
    def _test_expand(self, run_ipu=True):
        scope = fluid.core.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        with fluid.scope_guard(scope):
            with paddle.static.program_guard(main_prog, startup_prog):
                data = fluid.layers.fill_constant(
                    shape=[2, 3, 1], dtype="int32", value=1.0)
                out = fluid.layers.expand(data, [1, 2, 2])

                # data = fluid.layers.fill_constant(shape=[2, 3], dtype="int32", value=3)
                # expand_times = fluid.layers.fill_constant(shape=[2], dtype="int32", value=2)
                # out = fluid.layers.expand(data, expand_times=expand_times)

            if run_ipu:
                place = paddle.IPUPlace()
            else:
                place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(startup_prog)

            if run_ipu:
                feed_list = []
                fetch_list = [out.name]
                ipu_strategy = compiler.get_ipu_strategy()
                ipu_strategy.is_training = False
                program = compiler.IpuCompiler(
                    main_prog, ipu_strategy=ipu_strategy).compile(feed_list,
                                                                  fetch_list)
            else:
                program = main_prog

            result = exe.run(program, feed={}, fetch_list=[out])
            return result[0]

    def test_expand(self):
        cpu_res = self._test_expand(False)
        ipu_res = self._test_expand(True)

        self.assertTrue(np.allclose(ipu_res, cpu_res, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
