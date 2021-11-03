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

import paddle
import paddle.fluid.compiler as compiler
import paddle.optimizer
import paddle.static

paddle.enable_static()


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestCase(unittest.TestCase):
    def test_enable_patern(self):
        ipu_strategy = compiler.get_ipu_strategy()
        pattern = 'LSTMOp'
        # LSTMOp Pattern is not enabled by default
        # assert not ipu_strategy.is_pattern_enabled(pattern)
        ipu_strategy.enable_pattern(pattern)
        assert ipu_strategy.is_pattern_enabled(pattern) == True

    def test_disable_pattern(self):
        ipu_strategy = compiler.get_ipu_strategy()
        pattern = 'LSTMOp'
        ipu_strategy.enable_pattern(pattern)
        ipu_strategy.disable_pattern(pattern)
        assert ipu_strategy.is_pattern_enabled(pattern) == False


if __name__ == "__main__":
    unittest.main()
