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
import shutil

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.compiler as compiler
import paddle.optimizer
import paddle.static
from paddle.fluid.tests.unittests.ipu.op_test_ipu import IPUOpTest

paddle.enable_static()


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestBase(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_feed()
        self.set_attrs()

    def set_feed(self):
        self.feed_shape = []
        self.feed_shape.append([1, 3, 10, 10])

        self.feed_cpu = {}
        self.feed_ipu = {}
        data = np.random.uniform(size=self.feed_shape[0])
        self.feed_cpu["in_0"] = data.astype(np.float32)
        self.feed_ipu["in_0"] = data.astype(np.float16)

        self.feed_list = list(self.feed_cpu.keys())

    def set_attrs(self):
        self.attrs = {}
        self.attrs['steps'] = 100
        self.attrs['save_at_step'] = 20
        self.attrs['is_training'] = True
        self.attrs['opt_type'] = 'sgd'
        self.attrs['path'] = 'model'
        self.attrs['model_name'] = 'test'

    def _test_save(self):
        scope = fluid.core.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = self.SEED
        startup_prog.random_seed = self.SEED
        generator = fluid.unique_name.UniqueNameGenerator()
        self.full_name = '/'.join(
            [self.attrs['path'], self.attrs['model_name']])

        with fluid.unique_name.guard(generator):
            with fluid.scope_guard(scope):
                with paddle.static.program_guard(main_prog, startup_prog):
                    x = paddle.static.data(
                        name=self.feed_list[0],
                        shape=self.feed_shape[0],
                        dtype='float32')
                    scale = paddle.fluid.layers.scale(
                        x, scale=1.0, bias=0.0, bias_after_scale=True)
                    conv = paddle.static.nn.conv2d(
                        scale,
                        num_filters=3,
                        filter_size=3,
                        bias_attr=False,
                        name='conv2d')
                    loss = paddle.mean(conv)

                    if self.attrs['is_training']:
                        if self.attrs['opt_type'] == 'sgd':
                            sgd = paddle.optimizer.SGD(learning_rate=1e-2)
                            sgd.minimize(loss)
                        elif self.attrs['opt_type'] == 'adam':
                            adam = paddle.optimizer.Adam(learning_rate=1e-2)
                            adam.minimize(loss)
                        elif self.attrs['opt_type'] == 'lamb':
                            lamb = paddle.optimizer.Lamb(learning_rate=1e-2)
                            lamb.minimize(loss)
                    fetch_list = [loss.name]
                place = paddle.CPUPlace()
                exe = paddle.static.Executor(place)
                exe.run(startup_prog)

                result_cpu = []
                for i in range(self.attrs['steps']):
                    tmp = exe.run(main_prog,
                                  feed=self.feed_cpu,
                                  fetch_list=fetch_list)
                    result_cpu.append(tmp)
                place = paddle.IPUPlace()
                exe = paddle.static.Executor(place)
                exe.run(startup_prog)
                ipu_strategy = compiler.get_ipu_strategy()
                ipu_strategy.is_training = True
                ipu_strategy.enable_fp16 = True
                program = compiler.IpuCompiler(
                    main_prog, ipu_strategy=ipu_strategy).compile(
                        self.feed_list, fetch_list)

                result_ipu = []
                for i in range(self.attrs['steps']):
                    tmp = exe.run(program,
                                  feed=self.feed_ipu,
                                  fetch_list=fetch_list)
                    result_ipu.append(tmp)
                mae = np.mean(
                    np.abs(
                        np.asarray(result_cpu).flatten() - np.asarray(
                            result_ipu).flatten()))
                print('result_cpu:{} result_ipu:{} mae:{} '.format(
                    np.asarray(result_cpu).flatten(),
                    np.asarray(result_ipu).flatten(), mae))
                self.assertTrue(mae < 0.01)

                paddle.static.save_inference_model(
                    self.full_name, x, loss, exe, program=program.org_program)

    def _test_load(self, run_ipu):
        if run_ipu:
            place = paddle.IPUPlace()
        else:
            place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)

        [inference_program, feed_target_names, fetch_targets] = (
            paddle.static.load_inference_model(self.full_name, exe))

        feed = self.feed_ipu if run_ipu else self.feed_cpu
        if run_ipu:
            feed_list = feed_target_names
            fetch_list = [fetch_targets[0].name]
            ipu_strategy = compiler.get_ipu_strategy()
            ipu_strategy.is_training = False
            ipu_strategy.enable_fp16 = True
            program = compiler.IpuCompiler(
                inference_program,
                ipu_strategy=ipu_strategy).compile(feed_list, fetch_list)
        else:
            program = inference_program
        # feed fp32 temporiary
        tmp = exe.run(program, feed=feed, fetch_list=[fetch_targets])
        return tmp

    def test_base(self):
        self._test_save()
        cpu_res = self._test_load(False)
        ipu_res = self._test_load(True)

        mae = np.mean(
            np.abs(
                np.asarray(cpu_res).flatten() - np.asarray(ipu_res).flatten()))
        print('cpu_res:{} ipu_res:{} mae:{} '.format(cpu_res, ipu_res, mae))
        self.assertTrue(mae < 0.001)

        shutil.rmtree(self.attrs['path'], True)


if __name__ == "__main__":
    unittest.main()
