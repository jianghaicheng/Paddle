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

paddle.enable_static()


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestConfigure(unittest.TestCase):
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

        ipu_strategy.micro_batch_size = 4
        assert ipu_strategy.micro_batch_size == 4, "Set micro_batch_size Failed"

        ipu_strategy.is_training = False
        assert ipu_strategy.is_training == False, "Set is_training Failed"

        ipu_strategy.save_init_onnx = True
        assert ipu_strategy.save_init_onnx == True, "Set save_init_onnx Failed"

        ipu_strategy.save_onnx_checkpoint = True
        assert ipu_strategy.save_onnx_checkpoint == True, "Set save_onnx_checkpoint Failed"

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

        ipu_strategy.loss_scaling = 5.0
        assert ipu_strategy.loss_scaling == 5.0, \
            "Set loss_scaling Failed"

        ipu_strategy.max_weight_norm = 65504.0
        assert ipu_strategy.max_weight_norm == 65504.0, \
            "Set max_weight_norm Failed"

        assert ipu_strategy.enable_stochastic_rounding == False, \
            "Default value for enable_stochastic_rounding must be False"
        ipu_strategy.enable_stochastic_rounding = True
        assert ipu_strategy.enable_stochastic_rounding == True, \
            "Set enable_stochastic_rounding Failed"

        assert ipu_strategy.enable_fully_connected_pass == True, \
            "Default value for enable_fully_connected_pass must be False"
        ipu_strategy.enable_fully_connected_pass = False
        assert ipu_strategy.enable_fully_connected_pass == False, \
            "Set enable_fully_connected_pass Failed"

        assert ipu_strategy.enable_engine_caching == False, \
            "Default value for enable_engine_caching must be False"
        ipu_strategy.enable_engine_caching = True
        assert ipu_strategy.enable_engine_caching == True, \
            "Set enable_engine_caching Failed"

        assert ipu_strategy.cache_path == "session_cache", \
            "Default value for cache_path must be False"
        ipu_strategy.cache_path = "new_session_cache"
        assert ipu_strategy.cache_path == "new_session_cache", \
            "Set cache_path Failed"


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestEnablePattern(unittest.TestCase):
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


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestIpuStrategyLoadDict(unittest.TestCase):
    def test_enable_patern(self):
        ipu_strategy = compiler.get_ipu_strategy()
        test_conf = {
            "micro_batch_size": 23,
            "batches_per_step": 233,
            "enableGradientAccumulation": True,
            "enableReplicatedGraphs": True,
            "enable_fp16": True,
            "save_init_onnx": True,
            "save_onnx_checkpoint": True
        }
        ipu_strategy.load_dict(test_conf)
        for k, v in test_conf.items():
            assert v == getattr(ipu_strategy, k)


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestIpuStrategyEngineOptions(unittest.TestCase):
    def test_enable_patern(self):
        ipu_strategy = compiler.get_ipu_strategy()
        engine_conf = {
            'debug.allowOutOfMemory': 'true',
            'autoReport.directory': 'path',
            'autoReport.all': 'true'
        }
        ipu_strategy.engine_options = engine_conf
        for k, v in engine_conf.items():
            assert v == ipu_strategy.engine_options[k]


if __name__ == "__main__":
    unittest.main()
