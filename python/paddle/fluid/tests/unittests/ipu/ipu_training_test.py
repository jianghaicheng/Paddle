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

import paddle
import numpy as np

# 飞桨2.X默认模式为动态图，需要开启静态图模式
paddle.enable_static()

# 编译期：调用飞桨的API编写Python程序，如下述代码中定义了一个含conv2d的网络，并使用Adam优化器优化参数。
image = paddle.static.data(
    name='image', shape=[None, 3, 224, 224], dtype='float32')
conv_result = paddle.static.nn.conv2d(image, num_filters=64, filter_size=3)
loss = paddle.mean(conv_result)
adam = paddle.optimizer.Adam(learning_rate=1e-3)
adam.minimize(loss)

# 运行期：先运行一次startup program初始化网络参数，然后调用飞桨的Executor和CompiledProgram API运行网络。
place = paddle.IPUPlace()  # 使用何种设备运行网络，IPUPlace表示使用IPU运行
executor = paddle.static.Executor(place)  # 创建执行器
print("---------- startup_program --------------")
prog = paddle.static.default_startup_program()
print(prog._to_readable_code())
executor.run(
    paddle.static.default_startup_program())  # 运行startup program进行参数初始化
print("---------- main_program --------------")
prog = paddle.static.default_main_program()
print(prog._to_readable_code())

# 再使用CompiledProgram编译网络，准备执行。
compiled_program = paddle.static.CompiledProgram(
    paddle.static.default_main_program())

BATCH_NUM = 2
BATCH_SIZE = 32

for batch_id in range(BATCH_NUM):
    input_image = np.random.random([BATCH_SIZE, 3, 224, 224]).astype('float32')
    loss_numpy, = executor.run(
        compiled_program, feed={'image': input_image}, fetch_list=[loss])
    print("Batch {}, loss = {}".format(batch_id, loss_numpy))

# 关闭静态图模式
paddle.disable_static()
