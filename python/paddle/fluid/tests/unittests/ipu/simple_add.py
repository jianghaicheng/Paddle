# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle
import paddle.fluid.core as core
import paddle.fluid.compiler as compiler
from paddle.static import Program

paddle.enable_static()

a = paddle.static.data(name='a', shape=[1], dtype='float32')
b = paddle.static.data(name='b', shape=[1], dtype='float32')
c = a + b

place = paddle.IPUPlace()
executor = paddle.static.Executor(place)

print("------------------------")
print("default_startup_program:")
startup_prog = paddle.static.default_startup_program()
print(startup_prog._to_readable_code())
executor.run(startup_prog)

print("------------------------")
print("default_main_program:")
main_prog = paddle.static.default_main_program()
print(main_prog._to_readable_code())

feed_list = ['a', 'b']
fetch_list = ['tmp_0']
ipu_strategy = compiler.get_ipu_strategy()
ipu_strategy.is_training = False
ipu_strategy.num_ipus = 1

program = compiler.IpuCompiler(
    main_prog, ipu_strategy=ipu_strategy).compile(feed_list, fetch_list)
print("Program to run:")
print(program._to_readable_code())

result = executor.run(program,
                      feed={
                          'a': np.array(
                              [1], dtype=np.float32),
                          'b': np.array(
                              [1], dtype=np.float32)
                      },
                      fetch_list=[c])
print("result = {}".format(result))
