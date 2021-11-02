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
class TestConvNet(unittest.TestCase):
    def test_training(self):
        ipu_strategy = compiler.get_ipu_strategy()

        assert ipu_strategy.num_ipus == 1, "Default num_ipus must be 1"
        assert ipu_strategy.is_training == True, "Default is_training is True"
        assert ipu_strategy.enable_pipelining == False, \
            "Default enable_pipelining is False"
        assert ipu_strategy.enable_manual_shard == False, \
            "Default enable_manual_shard is False"

        ipu_strategy.num_ipus = 2
        assert ipu_strategy.num_ipus == 2, "Set num_ipus Failed"

        ipu_strategy.batches_per_step = 5
        assert ipu_strategy.batches_per_step == 5, \
            "Set batches_per_step Failed"

        ipu_strategy.batch_size = 4
        assert ipu_strategy.batch_size == 4, "Set batch_size Failed"

        ipu_strategy.is_training = False
        assert ipu_strategy.is_training == False, "Set is_training Failed"

        ipu_strategy.save_init_onnx = True
        assert ipu_strategy.save_init_onnx == True, "Set save_init_onnx Failed"

        ipu_strategy.save_last_onnx = True
        assert ipu_strategy.save_last_onnx == True, "Set save_last_onnx Failed"

        ipu_strategy.save_per_n_step = 10
        assert ipu_strategy.save_per_n_step == 10, "Set save_per_n_step Failed"

        ipu_strategy.need_avg_shard = True
        assert ipu_strategy.need_avg_shard == True, "Set need_avg_shard Failed"

        ipu_strategy.enable_fp16 = True
        assert ipu_strategy.enable_fp16 == True, "Set enable_fp16 Failed"

        ipu_strategy.enable_pipelining = True
        assert ipu_strategy.enable_pipelining == True, \
            "Set enable_pipelining Failed"

        ipu_strategy.enable_manual_shard = True
        assert ipu_strategy.enable_manual_shard == True, \
            "Set enable_manual_shard Failed"

        ipu_strategy.enable_half_partial = True
        assert ipu_strategy.enable_half_partial == True, \
            "Set enable_half_partial Failed"

        ipu_strategy.available_mem_proportion = 0.5
        assert ipu_strategy.available_mem_proportion == 0.5, \
            "Set available_mem_proportion Failed"


if __name__ == "__main__":
    unittest.main()
