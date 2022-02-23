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

import unittest

import numpy as np
import paddle
import paddle.static
from paddle.fluid.tests.unittests.ipu.op_test_ipu import IPUOpTest


@unittest.skipIf(True, "disable for lack of IPU")
class TestBase(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_training()
        self.set_op_attr()
        self.set_data_feed()

    def set_training(self):
        self.is_training = False
        self.epoch = 10

    def set_op_attr(self):
        self.attrs = {
            "batches_per_step": 1,
            "enable_pipeline": False,
            "enableGradientAccumulation": False,
            "accumulation_factor": 1,
            "enableReplicatedGraphs": False,
            "replicatedGraphCount": 1,
            "cpu_bs": 1,
            "ipu_bs": 1
        }

    def set_data_feed(self):
        np_image = np.random.rand(1, 3, 10, 10).astype(np.float32)
        self.feed_cpu = {"image": np_image}
        self.feed_ipu = {"image": np_image}

    def _test_base(self, run_ipu=True):
        scope = paddle.static.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = self.SEED
        startup_prog.random_seed = self.SEED

        bs = self.attrs['ipu_bs'] if run_ipu else self.attrs['cpu_bs']
        with paddle.static.scope_guard(scope):
            with paddle.static.program_guard(main_prog, startup_prog):
                image = paddle.static.data(
                    name='image', shape=[bs, 3, 10, 10], dtype='float32')
                with paddle.static.ipu_shard_guard(index=0):
                    conv1 = paddle.static.nn.conv2d(
                        image, num_filters=3, filter_size=3, bias_attr=False)
                with paddle.static.ipu_shard_guard(index=1):
                    conv2 = paddle.static.nn.conv2d(
                        conv1, num_filters=3, filter_size=3, bias_attr=False)
                    # should consider influence of bs
                    loss = paddle.mean(conv2)

                if self.is_training:
                    opt = None
                    if self.attrs['optimizer'] == 'sgd':
                        opt = paddle.optimizer.SGD(learning_rate=1e-2)
                    elif self.attrs['optimizer'] == 'adam':
                        opt = paddle.optimizer.Adam(learning_rate=1e-2)
                    elif self.attrs['optimizer'] == 'lamb':
                        opt = paddle.optimizer.Lamb(learning_rate=1e-2)
                    if opt is not None:
                        opt.minimize(loss)

            if run_ipu:
                place = paddle.IPUPlace()
            else:
                place = paddle.CPUPlace()
            executor = paddle.static.Executor(place)
            executor.run(startup_prog)

            if run_ipu:
                feed_list = [image.name]
                fetch_list = [loss.name]
                ipu_strategy = paddle.static.IpuStrategy()
                ipu_strategy.enableReplicatedGraphs = self.attrs[
                    'enableReplicatedGraphs']
                ipu_strategy.replicatedGraphCount = self.attrs[
                    'replicatedGraphCount']
                ipu_strategy.num_ipus = 2 * self.attrs['replicatedGraphCount']
                ipu_strategy.set_graph_config(is_training=self.is_training)
                ipu_strategy.enable_manual_shard = True
                ipu_strategy.enable_pipelining = self.attrs['enable_pipeline']
                ipu_strategy.batches_per_step = self.attrs['batches_per_step']
                ipu_strategy.enableGradientAccumulation = self.attrs[
                    'enableGradientAccumulation']
                ipu_strategy.accumulationFactor = self.attrs[
                    'accumulationFactor']
                program = paddle.static.IpuCompiledProgram(
                    main_prog,
                    ipu_strategy=ipu_strategy).compile(feed_list, fetch_list)
            else:
                program = main_prog

            feed = self.feed_ipu if run_ipu else self.feed_cpu
            epoch = self.epoch
            if not run_ipu:
                epoch *= self.attrs['replicatedGraphCount']
                epoch *= self.attrs['batches_per_step']
                epoch *= self.attrs['accumulationFactor']
                epoch = epoch / (self.attrs['cpu_bs'] / self.attrs['ipu_bs'])
            result = []
            for i in range(int(epoch)):
                loss_res = executor.run(program, feed=feed, fetch_list=[loss])
                result.append(loss_res)
            return np.array(result).flatten()

    def test(self):
        cpu_outputs = self._test_base(False)
        ipu_outputs = self._test_base(True)

        self.assertTrue(np.allclose(cpu_outputs, ipu_outputs, atol=self.atol))


class TestReplicaInference(TestBase):
    def set_op_attr(self):
        self.attrs = {
            "batches_per_step": 1,
            "enable_pipeline": False,
            "enableGradientAccumulation": False,
            "accumulationFactor": 1,
            "enableReplicatedGraphs": True,
            "replicatedGraphCount": 2,
            "cpu_bs": 1,
            "ipu_bs": 1
        }

    def set_data_feed(self):
        np_image = np.random.rand(1, 3, 10, 10).astype(np.float32)
        self.feed_cpu = {"image": np_image}
        self.feed_ipu = {
            "image": np.tile(np_image,
                             [self.attrs['replicatedGraphCount'], 1, 1, 1])
        }


class TestPipelineInference(TestBase):
    def set_op_attr(self):
        self.attrs = {
            "batches_per_step": 2,
            "enable_pipeline": True,
            "enableGradientAccumulation": False,
            "accumulationFactor": 1,
            "enableReplicatedGraphs": False,
            "replicatedGraphCount": 1,
            "cpu_bs": 1,
            "ipu_bs": 1
        }

    def set_data_feed(self):
        np_image = np.random.rand(1, 3, 10, 10).astype(np.float32)
        self.feed_cpu = {"image": np_image}
        self.feed_ipu = {
            "image": np.tile(np_image,
                             [self.attrs['batches_per_step'], 1, 1, 1])
        }


class TestTrainBase(TestBase):
    def set_training(self):
        self.is_training = True
        self.epoch = 10

    def set_op_attr(self):
        self.attrs = {
            "batches_per_step": 1,
            "enable_pipeline": False,
            "enableGradientAccumulation": False,
            "accumulationFactor": 1,
            "enableReplicatedGraphs": False,
            "replicatedGraphCount": 1,
            "optimizer": 'sgd',
            "cpu_bs": 1,
            "ipu_bs": 1
        }


class TestReplicaTrain(TestTrainBase):
    def set_op_attr(self):
        self.attrs = {
            "batches_per_step": 1,
            "enable_pipeline": False,
            "enableGradientAccumulation": False,
            "accumulationFactor": 1,
            "enableReplicatedGraphs": True,
            "replicatedGraphCount": 2,
            "optimizer": 'sgd',
            "cpu_bs": 2,
            "ipu_bs": 1
        }

    def set_data_feed(self):
        np_image = np.random.rand(1, 3, 10, 10).astype(np.float32)
        self.feed_cpu = {
            "image": np.tile(np_image, [self.attrs['cpu_bs'], 1, 1, 1])
        }
        self.feed_ipu = {
            "image": np.tile(np_image,
                             [self.attrs['replicatedGraphCount'], 1, 1, 1])
        }

    def test(self):
        cpu_outputs = self._test_base(False)
        ipu_outputs = self._test_base(True)[::2]

        self.assertTrue(np.allclose(cpu_outputs, ipu_outputs, atol=self.atol))


class TestPipelineTrain(TestTrainBase):
    def set_op_attr(self):
        self.attrs = {
            "batches_per_step": 3,
            "enable_pipeline": True,
            "enableGradientAccumulation": True,
            "accumulationFactor": 3,
            "enableReplicatedGraphs": False,
            "replicatedGraphCount": 1,
            "optimizer": 'sgd',
            "cpu_bs": 3,
            "ipu_bs": 1
        }

    def set_data_feed(self):
        np_image = np.random.rand(1, 3, 10, 10).astype(np.float32)
        self.feed_cpu = {
            "image": np.tile(np_image, [self.attrs['cpu_bs'], 1, 1, 1])
        }
        bps_acc = self.attrs['batches_per_step'] * self.attrs[
            'accumulationFactor']
        self.feed_ipu = {"image": np.tile(np_image, [bps_acc, 1, 1, 1])}

    def test(self):
        cpu_outputs = self._test_base(False)
        ipu_outputs = self._test_base(True)[::3]

        self.assertTrue(np.allclose(cpu_outputs, ipu_outputs, atol=self.atol))


class TestAdamTrain(TestTrainBase):
    def set_op_attr(self):
        self.attrs = {
            "batches_per_step": 1,
            "enable_pipeline": False,
            "enableGradientAccumulation": False,
            "accumulationFactor": 1,
            "enableReplicatedGraphs": False,
            "replicatedGraphCount": 1,
            "optimizer": 'adam',
            "cpu_bs": 1,
            "ipu_bs": 1
        }


class TestAdamReplicaTrain(TestReplicaTrain):
    def set_op_attr(self):
        self.attrs = {
            "batches_per_step": 1,
            "enable_pipeline": False,
            "enableGradientAccumulation": False,
            "accumulationFactor": 1,
            "enableReplicatedGraphs": True,
            "replicatedGraphCount": 2,
            "optimizer": 'adam',
            "cpu_bs": 2,
            "ipu_bs": 1
        }


class TestAdamPipelineTrain(TestPipelineTrain):
    def set_op_attr(self):
        self.attrs = {
            "batches_per_step": 3,
            "enable_pipeline": True,
            "enableGradientAccumulation": True,
            "accumulationFactor": 3,
            "enableReplicatedGraphs": False,
            "replicatedGraphCount": 1,
            "optimizer": 'adam',
            "cpu_bs": 3,
            "ipu_bs": 1
        }


class TestAdamRecomputationTrain(TestPipelineTrain):
    def set_op_attr(self):
        self.attrs = {
            "batches_per_step": 3,
            "enable_pipeline": True,
            "enableGradientAccumulation": True,
            "accumulationFactor": 3,
            "enableReplicatedGraphs": False,
            "replicatedGraphCount": 1,
            "optimizer": 'adam',
            "auto_recomputation": 3,
            "cpu_bs": 3,
            "ipu_bs": 1
        }


@unittest.skip('skip TestLambTrain')
class TestLambTrain(TestAdamTrain):
    def set_op_attr(self):
        self.attrs = {
            "batches_per_step": 1,
            "enable_pipeline": False,
            "enableGradientAccumulation": False,
            "accumulationFactor": 1,
            "enableReplicatedGraphs": False,
            "replicatedGraphCount": 1,
            "optimizer": 'lamb',
            "cpu_bs": 1,
            "ipu_bs": 1
        }


@unittest.skip('skip TestLambReplicaTrain')
class TestLambReplicaTrain(TestAdamReplicaTrain):
    def set_op_attr(self):
        self.attrs = {
            "batches_per_step": 1,
            "enable_pipeline": False,
            "enableGradientAccumulation": False,
            "accumulationFactor": 1,
            "enableReplicatedGraphs": True,
            "replicatedGraphCount": 2,
            "optimizer": 'lamb',
            "cpu_bs": 2,
            "ipu_bs": 1
        }


@unittest.skip('skip TestLambPipelineTrain')
class TestLambPipelineTrain(TestAdamPipelineTrain):
    def set_op_attr(self):
        self.attrs = {
            "batches_per_step": 3,
            "enable_pipeline": True,
            "enableGradientAccumulation": True,
            "accumulationFactor": 3,
            "enableReplicatedGraphs": False,
            "replicatedGraphCount": 1,
            "optimizer": 'lamb',
            "cpu_bs": 3,
            "ipu_bs": 1
        }


if __name__ == "__main__":
    unittest.main()
