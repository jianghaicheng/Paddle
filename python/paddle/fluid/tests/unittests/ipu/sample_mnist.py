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

import sys

import numpy as np
import paddle
import paddle.static
import paddle.fluid as fluid
# IPU graph compiler
import paddle.fluid.compiler as compiler

paddle.enable_static()

BATCH_SIZE = 32


# Accuracy Op
def numeric_topk(ndarray, k=1):
    values_cpy_or_ref = -ndarray
    sorted_indices = np.argpartition(values_cpy_or_ref, k, axis=1)
    row_indices = np.arange(ndarray.shape[0])[:, None]
    topk_indices = sorted_indices[:, 0:k]
    topk_values = ndarray[row_indices, topk_indices]
    return topk_values, topk_indices


def topKAccuracy(pred, labels, k=1):
    shape = pred.shape
    _, topk_indices = numeric_topk(pred, k=k)

    num_correct = 0
    for i in range(shape[0]):
        for j in range(k):
            if topk_indices[i, j] == labels[i]:
                num_correct += 1
                break

    accuracy = (num_correct * 1.) / shape[0]
    return accuracy


# Build model
class MNIST:
    def __init__(self):
        self.mode = 'train'
        self.optimizer = None
        self.batch_size = BATCH_SIZE

    def build_model(self):
        self._build_input()
        self._build_body()
        return self._outputs

    def _build_input(self):
        self.img = paddle.static.data(
            name='img', shape=[self.batch_size, 1, 28, 28], dtype='float32')
        self.label = paddle.static.data(
            name='label', shape=[self.batch_size, 1], dtype='int32')

    def _build_body(self):
        conv_pool_1 = fluid.nets.simple_img_conv_pool(
            input=self.img,
            num_filters=20,
            filter_size=5,
            pool_size=2,
            pool_stride=2,
            act="relu")
        conv_pool_bn_1 = fluid.layers.batch_norm(conv_pool_1)
        conv_pool_2 = fluid.nets.simple_img_conv_pool(
            input=conv_pool_bn_1,
            num_filters=50,
            filter_size=5,
            pool_size=2,
            pool_stride=2,
            act="relu")
        self.featmap = [conv_pool_2]
        self._loss()

    def _loss(self):
        prediction = fluid.layers.fc(
            input=self.featmap[0], size=10, act='softmax')
        loss = fluid.layers.cross_entropy(input=prediction, label=self.label)
        avg_loss = fluid.layers.mean(loss)
        self._outputs = [prediction, avg_loss]

    @property
    def inputs(self):
        return [self.img, self.label]

    @property
    def outputs(self):
        return self._outputs


# Mnist dataset
class MnistDataset(paddle.vision.datasets.MNIST):
    def __init__(self, mode):
        super(MnistDataset, self).__init__(mode=mode)

    def __getitem__(self, idx):
        img = np.reshape(self.images[idx], [1, 28, 28])
        img = img / 255.0 * 2.0 - 1.0
        # IPU only support INT32 Label in Cross_Entropy Op 
        return img.astype('float32'), np.array(self.labels[idx]).astype(
            np.int32)

    def __len__(self):
        return len(self.images)


# Run training
def train(exec, model, train_reader, train_program):
    for data in train_reader():
        exec.run(train_program,
                 feed={
                     model.inputs[0].name: data[0],
                     model.inputs[1].name: data[1]
                 },
                 fetch_list=model.outputs[1])


# Run validation
def test(exec, model, test_reader, validation_program):
    validation_acc_set = []
    for test_data in test_reader():
        metrics = exec.run(program=validation_program,
                           feed={
                               model.inputs[0].name: test_data[0],
                               model.inputs[1].name: test_data[1]
                           },
                           fetch_list=[model.outputs[0], model.outputs[1]])
        pred = metrics[0]
        labels = test_data[1]
        accuracy = topKAccuracy(pred, labels, 1)
        validation_acc_set.append(float(accuracy))
        validation_avg_acc = np.array(validation_acc_set).mean()
        print("Validation average accuracy:" + str(validation_avg_acc))


def main():
    startup_program = paddle.static.Program()
    main_program = paddle.static.Program()
    place = paddle.IPUPlace()
    exe = paddle.static.Executor(place)

    train_reader = paddle.io.DataLoader(
        MnistDataset(mode='train'), batch_size=BATCH_SIZE, drop_last=True)
    test_reader = paddle.io.DataLoader(
        MnistDataset(mode='test'), batch_size=BATCH_SIZE, drop_last=True)

    mnist = MNIST()

    with paddle.static.program_guard(main_program, startup_program):
        with paddle.utils.unique_name.guard():
            pred, avg_loss = mnist.build_model()
            validation_program = main_program.clone(for_test=True)
            optimizer = fluid.optimizer.Adam(learning_rate=0.001)
            optimizer.minimize(avg_loss)
            exe.run(startup_program)

            feed_list = [inp.name for inp in mnist.inputs]
            fetch_list = [out.name for out in mnist.outputs]
            # Build training program with IPU
            ipu_strategy = compiler.get_ipu_strategy()  # Training IPU strategy
            ipu_strategy.num_ipus = 1  # The number of IPU
            ipu_strategy.is_training = True  # Training or Inference
            ipu_compiler = compiler.IpuCompiler(
                main_program, ipu_strategy=ipu_strategy)
            main_program = ipu_compiler.compile(feed_list, fetch_list)
            train(exe, mnist, train_reader, main_program)

            # Build validation program with IPU
            ipu_strategy = compiler.get_ipu_strategy()  # Validation IPU strategy
            ipu_strategy.num_ipus = 1
            ipu_strategy.is_training = False
            ipu_compiler = compiler.IpuCompiler(
                validation_program, ipu_strategy=ipu_strategy)
            validation_program = ipu_compiler.compile(feed_list, fetch_list)
            test(exe, mnist, test_reader, validation_program)
    return 0


if __name__ == "__main__":
    sys.exit(main())
