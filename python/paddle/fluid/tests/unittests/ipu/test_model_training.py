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
import paddle.static
import paddle.fluid.compiler as compiler

paddle.seed(2021)
np.random.seed(2021)

if __name__ == "__main__":
    run_on_ipu = True
    fetch_loss = True

    paddle.enable_static()

    # model input
    image = paddle.static.data(
        name='image', shape=[1, 3, 10, 10], dtype='float32')
    conv1 = paddle.static.nn.conv2d(
        image, num_filters=3, filter_size=3, bias_attr=False)
    conv2 = conv1 + conv1
    loss = paddle.mean(conv2)
    adam = paddle.optimizer.Adam(learning_rate=1e-3)

    # apply optimizer
    adam.minimize(loss)

    # switch cpu/ipu place
    if run_on_ipu:
        place = paddle.IPUPlace()
    else:
        place = paddle.CPUPlace()
    executor = paddle.static.Executor(place)

    startup_prog = paddle.static.default_startup_program()
    executor.run(startup_prog)

    # graph
    feed_list = [image.name]

    # switch loss and conv1
    if fetch_loss:
        fetch_node = loss
    else:
        fetch_node = conv1
    fetch_list = [fetch_node.name]

    main_prog = paddle.static.default_main_program()

    if run_on_ipu:
        ipu_strategy = compiler.get_ipu_strategy()
        ipu_strategy.is_training = False  # default True
        program = compiler.IpuCompiler(
            main_prog, ipu_strategy=ipu_strategy).compile(feed_list, fetch_list)
    else:
        program = main_prog

    np_image = np.random.rand(1, 3, 10, 10).astype(np.float32)
    res = executor.run(program,
                       feed={image.name: np_image},
                       fetch_list=[fetch_node])

    print(res)
