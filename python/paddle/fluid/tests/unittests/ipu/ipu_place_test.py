#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

"""simple test for ipu_place.

Args:
    place (IPUPlace The place where the op runs.
Returns:
    None.
"""

import paddle
import numpy as np
import os
import paddle.fluid.layers as layers
import faulthandler;

faulthandler.enable()
os.environ['TF_POPLAR_FLAGS']="--use_cpu_model”"

# 飞桨2.X默认模式为动态图，需要开启静态图模式
paddle.enable_static()

# 编译期：调用飞桨的API编写Python程序，如下述代码中定义了一个含conv2d的网络，并使用Adam优化器优化参数。
a = paddle.static.data(name='a', shape=[1], dtype='float32')
b = paddle.static.data(name='b', shape=[1], dtype='float32')
c = a + b
# paddle.assign_value(a,b,c)
# 运行期：先运行一次startup program初始化网络参数，然后调用飞桨的Executor和CompiledProgram API运行网络。
place = paddle.IPUPlace(0) # 使用何种设备运行网络，IPUPlace表示使用IPU运行，IUDAPlace表示使用IPU运行

# x = paddle.ones([2,2])
# y = paddle.ones([2,2])
# input = paddle.ones([2,2])
# out = paddle.gctest( input=input, x=x, y=y, beta=0.5, alpha=5.0 )
# print("output values is ",out,x)

# d = paddle.gctest(input =a,x=a, y=b)

executor = paddle.static.Executor(place) # 创建执行器
# executor.run(paddle.static.default_startup_program()) # 运行startup program进行参数初始化

prog = paddle.static.default_main_program()
print(prog._to_readable_code())

# 再使用CompiledProgram编译网络，准备执行。
# compiled_program = paddle.static.CompiledProgram(paddle.static.default_main_program())

result = executor.run(prog, feed={'a': np.array([100], dtype=np.float32), 'b': np.array([200], dtype=np.float32)}, fetch_list=[c])
print("result = {}".format(result))

# 关闭静态图模式
paddle.disable_static()
