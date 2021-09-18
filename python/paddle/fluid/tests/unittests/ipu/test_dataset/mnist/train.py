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

# refrenece : https://github.com/PaddlePaddle/models/blob/develop/dygraph/mnist/train.py (Paddle >= 2.1)
#             https://github.com/PaddlePaddle/book/blob/develop/02.recognize_digits/README.cn.md (Paddle <= 1.6)
#             https://github.com/graphcore/tutorials/tree/sdk-release-2.1/simple_applications/popart/mnist (IPU Popart framework example)

import os
import sys

sys.path.append("..")

import copy
import argparse
import ast
from contextlib import contextmanager
from functools import partial

import numpy as np
from PIL import Image
import paddle
import paddle.static
import paddle.fluid as fluid
from paddle.fluid import core
import paddle.fluid.framework as framework
# IPU graph compiler
import paddle.fluid.compiler as compiler
from paddle.vision.transforms.transforms import SaturationTransform
from model import MNIST
from mnist import set_random_seed
from logger import setup_logger

paddle.enable_static()

logger = setup_logger('mnist:trainer')

# define batch size
# 
BATCH_SIZE = 64
# TODO(yiakwy)
BATCHES_PER_STEP=-1

# DEVICE_SUFFIX="ipu"

# seed for continuous evaluation
SEED = 90

def parse_args():
    parser = argparse.ArgumentParser("Training mnist on IPUs", 
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--enable_ce",
        action='store_true',
        help="If set, run the task with continuous evaluation logs.")
    parser.add_argument(
        "--use_data_parallel",
        type=ast.literal_eval,
        default=False,
        help="The flag indicating whether to use data parallel mode to train the model."
    )
    parser.add_argument(
        "--use_ipu",
        type=bool,
        default=False,
        help="Whether to use IPU or not.")
    parser.add_argument(
        "--num_ipus",
        type=int, 
        default=1, 
        help="Number of ipus"
    )
    parser.add_argument(
        "--no_pipelining",
        action="store_true",
        help="If set, shards of Graph on different IPUs will not be pipelined."
    )
    parser.add_argument(
        "--replication-factor",
        type=int,
        default=1,
        help="Number of times to replicate the graph to perform data parallel"
             " training. Must be a factor of the number of IPUs."
    )
    parser.add_argument(
        "--draw_ir_graph",
        action="store_false",
        help="draw IR graph for debug"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=5, help="number of epochs.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="log_dir/",
        help="output directory"
    )
    args = parser.parse_args()
    return args


def draw_ir_graph(prog, name='raw_graph', save_path="", base_path="log_dir/ir"):
    if prog._graph is None:
        logger.info("IRGraph is not created!")
        graph = framework.IrGraph(core.Graph(prog.desc), for_test=True)
    else:
        graph = prog._graph
    
    save_path = os.path.join(base_path, save_path)
    # call graph_viz_pass
    graph.draw(save_path, name)


# Paddle >= 2.1, use paddle.vision.datasets.MNIST
# Paddle <= 1.6, use paddle.dataset.mnist, deprecated in Paddle >= 2.1 
class MnistDataset(paddle.vision.datasets.MNIST):
    def __init__(self, mode, return_label=True):
        super(MnistDataset, self).__init__(mode=mode)
        self.return_label = return_label

    def __getitem__(self, idx):
        img = np.reshape(self.images[idx], [1, 28, 28])
        img = img / 255.0 * 2.0 - 1.0
        if self.return_label:
            return img.astype('float32'), np.array(self.labels[idx]).astype(np.int64) # INT64 is not supported in Poplar and Popart framework
        return img,

    def __len__(self):
        return len(self.images)


def topKAccuracy(pred, labels, k=1):
    """
    Args:
        pred: prediction numpy array, shape=(M,N) 
        labels: ground truth, shape(M,))
        k: index with kth largeast probability
    Returns:
        float: topK accuracy value
    """
    if not isinstance(pred, np.ndarray):
        raise ValueError("pred must be a numpy array!")
    shape = pred.shape
    if len(shape) != 2:
        raise ValueError("Wrong dimension for prediction, expected 2 but found %d" % len(shape))
    sorted_indices = np.argpartition(-pred, k, axis=1)
    # row_indices = np.arange(pred.shape[0])[:,None]
    # topk_indices_cols = np.argsort(pred[row_indices, sorted_indices[:, 0:k]], axis=1)
    # topk_indices = sorted_indices[:,0:k][row_indices, topk_indices_cols]
    topk_indices = sorted_indices[:,0:k]

    num_correct = 0
    for i in range(shape[0]):
        for j in range(k):
            if topk_indices[i,j] == labels[i]:
                num_correct += 1
                break

    accuracy = (num_correct * 1.) / shape[0]
    return accuracy



def train(epochs, exec, model, feeder, save_dir, 
          train_reader, train_program,
          test_reader, validation_program):
    report = []
    step = 0
    for pass_id in range(epochs):
        # test for epoch
        validation_loss = -1
        if pass_id > 0:#(pass_id+1) % 10 == 0:
            validation_loss_set = []
            validation_acc_set = []
            # TODO(yiakwy) : do evaluation
            for test_data in test_reader():
                if model.use_topK_Accuracy_op:
                    metrics = exec.run(program=validation_program,
                    feed={inp.name : test_data[i] for i, inp in enumerate(model.inputs)},
                    fetch_list=model.outputs[1:3])

                    validation_loss_set.append(float(metrics[0]))
                    validation_acc_set.append(float(metrics[1]))
                else:
                    if model.cfg.get("use_ipu", False):
                        test_data = [np.array(data[0]), np.array(data[1]).astype('int32')]
                    metrics = exec.run(program=validation_program,
                    feed={inp.name : test_data[i] for i, inp in enumerate(model.inputs)},
                    fetch_list=[model.outputs[0], model.outputs[1], model.inputs[1]])

                    # TODO(yiakwy) : calculate topK accuracy with numpy
                    pred = metrics[0]
                    avg_loss = metrics[1]
                    labels = metrics[2]

                    accuracy = topKAccuracy(pred, labels, 1)

                    # add to validation set
                    validation_loss_set.append(float(avg_loss))
                    validation_acc_set.append(float(accuracy))

            # check average loss value
            validation_avg_loss = np.array(validation_loss_set).mean()
            validation_avg_acc = np.array(validation_acc_set).mean()
            
            # check exitance condition
            validation_loss = validation_avg_loss

            # processing logics
            report.append((pass_id, validation_avg_loss, validation_avg_acc))
        for batch_id, data in enumerate(train_reader()):
            # TODO(yiakwy) : feeder.feed(data) does not work
            # metrics = exec.run(train_program, feed=feeder.feed(data), 
            # fetch_list=model.outputs[1:3])
            metrics = exec.run(train_program, 
            feed={inp.name : data[i] for i, inp in enumerate(model.inputs)},
            fetch_list=model.outputs[1])
            if batch_id % 100 == 0:
                if validation_loss > 0:
                    print("Epoch %d, batch %d, Cost %f, Validation Cost %f" % (
                        pass_id, batch_id, metrics[0], validation_loss
                    ))
                else:
                    print("Epoch %d, batch %d, Cost %f" % (
                        pass_id, batch_id, metrics[0]
                    ))

            step += 1

        if save_dir is not None:
            if True:#not model.cfg.get("use_ipu", False):
                # TODO(yiak) : does not work in IPU
                paddle.static.save_inference_model(
                    save_dir+"recognize_digits_%s" % model.cfg.get("device_suffix", "cpu"),
                    model.inputs[0], model.outputs[0],
                    exec, program=train_program
                )
            else:
                paddle.static.save(train_program, save_dir+"recognize_digits_%s_test" % model.cfg.get("device_suffix", "ipu"))

    # find the best pass
    best = sorted(report,key=lambda record: float(record[1]))[0]
    print('Best pass is %s, validation average loss is %s' % (best[0], best[1]))
    print('The classification accuracy is %.2f%%' % (float(best[2]) * 100))


def main():
    FLAGS = parse_args()

    # set device for static graph
    startup_program = paddle.static.Program()
    train_program = paddle.static.Program()

    if not FLAGS.use_ipu:
        if not FLAGS.use_data_parallel:
            # TODO(yiakwy) : 
            # place = paddle.CPUPlace(0) does not work
            place = paddle.CPUPlace()
        else:
            place = paddle.CPUPlace()
    else:
        place = paddle.IPUPlace()
    executor = paddle.static.Executor(place)

    if FLAGS.enable_ce:
        set_random_seed(SEED)
        startup_program.random_seed = SEED
        train_program.random_seed = SEED

    if FLAGS.use_data_parallel:
        # TODO(yiakwy) : 
        #   1. use paddle.static.ParallelExecutor to train with CPU(IPU?) devices
        #   2. use paddle.DataParallel to dispatch data to CPU(IPU?) devices
        # see https://www.paddlepaddle.org.cn/documentation/docs/zh/1.6/api_guides/low_level/parallel_executor.html
        raise NotImplemented("not implemented yet!")

    train_exc = executor

    # create data reader
    logger.info("Loading data ...")
    if not FLAGS.use_data_parallel:
        train_reader = paddle.io.DataLoader(
            MnistDataset(mode='train'), batch_size=BATCH_SIZE, drop_last=True)
        
        # used for evaluation while training
        test_reader = paddle.io.DataLoader(
            MnistDataset(mode='test'), batch_size=BATCH_SIZE, drop_last=True)
    else:
        train_sampler = paddle.io.DistributeBatchSampler(
            MnistDataset(mode='train'), batch_size=BATCH_SIZE, drop_last=True)
        train_reader = paddle.io.DataLoader(
            MnistDataset(mode='train'), batch_sampler=train_sampler)

        # used for evaluation while training
        test_sampler = paddle.io.DistributeBatchSampler(
            MnistDataset(mode='test'), batch_size=BATCH_SIZE, drop_last=True)
        test_reader = paddle.io.DataLoader(
            MnistDataset(mode='test'), batch_sampler=test_sampler)
    logger.info("Data loaded.")

    epochs = FLAGS.num_epochs
    save_dir = FLAGS.save_dir
    num_ipus = FLAGS.num_ipus
    enable_pipelining = not FLAGS.no_pipelining
    will_draw_ir_graph = FLAGS.draw_ir_graph
    device_suffix = "ipu" if FLAGS.use_ipu else "cpu"

    # create config
    cfg = {}
    cfg["batch_size"] = BATCH_SIZE
    cfg["use_ipu"] = FLAGS.use_ipu
    cfg["device_suffix"] = device_suffix

    # create model
    mnist = MNIST(cfg)

    # add model
    with paddle.static.program_guard(main_program=train_program, startup_program=startup_program):
        logger.info("Constructing the computation graph ...")
        mnist.build_model()
        logger.info("Computation graph built.")
        # initailize the graph
        train_exc.run(startup_program)
        validation_program = train_program.clone(for_test=True)

        if FLAGS.use_ipu:
            # Pipeline with tensorflow frontend: https://docs.graphcore.ai/projects/tensorflow1-user-guide/en/latest/perf_training.html#pipelined-training
            ipu_strategy = compiler.get_ipu_strategy()
            ipu_strategy.num_ipus = num_ipus
            ipu_strategy.enable_pipelining = enable_pipelining
            ipu_compiler = compiler.IpuCompiler(
                train_program, ipu_strategy=ipu_strategy)
            if will_draw_ir_graph:
                logger.info("Drawing IR graph ...")
                draw_ir_graph(train_program)
                logger.info("Complete drawing.")
            logger.info("Compiling graph on IPU devices ...")
            feed_list = [inp.name for inp in mnist.inputs]
            fetch_list = [out.name for out in mnist.outputs]
            train_program = ipu_compiler.compile(feed_list, fetch_list)
            logger.info("Complete compiling.")
            # Only forward graph contained in the new train_program
            # TODO(yiakwy) : add support to create `popart.InferenceSession` in python frontend
            validation_program = train_program.clone(for_test=False)
        else:
            if will_draw_ir_graph:
                logger.info("Drawing IR graph ...")
                draw_ir_graph(train_program)
                logger.info("Complete drawing.")


        # create data feader
        # TODO(yiak): This does not work for LoDTensor
        # feeder = fluid.DataFeeder(feed_list=mnist.inputs, place=place)
        feeder = None

        # train model
        train(epochs, 
              train_exc, 
              mnist, 
              feeder, # deprecated
              save_dir, 
              train_reader, train_program, 
              test_reader, validation_program)

    return 0

if __name__ == "__main__":
    sys.exit(main())