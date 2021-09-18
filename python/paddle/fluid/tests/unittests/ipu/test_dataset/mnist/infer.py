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
from logger import setup_logger

paddle.enable_static()

logger = setup_logger('mnist:infer')

DEVICE_SUFFIX=""

def parse_args():
    parser = argparse.ArgumentParser("Training mnist on IPUs", 
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--img",
        type=str,
        default="",
        help="image path"
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

def apply_pseudo_batch_size_pass(prog, batch_size, var_name):
    # transform feed var batch_size to 1
    global_block = prog.global_block()
    if var_name in global_block.vars:
        feed_var = global_block.vars[var_name] # Call API
        # modify attrs
        # TODO(yiakwy) : hard coded
        feed_var.desc.set_shape([1,1,28,28])
        logger.info("Change batch size of var %s from %d to %d")
        return

    raise ValueError("Cannot find variable %s in the program description" % var_name)

def load_image(file):
    im = Image.open(file).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    im = im / 255.0 * 2.0 - 1.0
    return im

def infer(exec, infer_prog, img, feed_name_list, fetch_list):
    preds = exec.run(
        infer_prog,
        feed={feed_name_list[0]: img},
        fetch_list=fetch_list)
    sorted_preds = np.argsort(preds)
    print("digit hand write number picture is recognized as : %d" % 
    sorted_preds[0][0][-1])


def main():
    FLAGS = parse_args()

    # set device for static graph
    if not FLAGS.use_ipu:
        DEVICE_SUFFIX = "cpu"
        place = paddle.CPUPlace()
    else:
        DEVICE_SUFFIX = "ipu"
        place = paddle.IPUPlace()
    executor = paddle.static.Executor(place)

    infer_exc = executor

    # Reading images
    logger.info("Reading data ...")
    pwd = os.path.dirname(os.path.realpath(__file__))
    img = load_image(os.path.join(pwd,FLAGS.img))
    logger.info("Complete reading image %s" % FLAGS.img)

    save_dir = FLAGS.save_dir
    num_ipus = FLAGS.num_ipus
    enable_pipelining = not FLAGS.no_pipelining
    will_draw_ir_graph = FLAGS.draw_ir_graph

    # add model
    infer_scope = paddle.static.Scope()
    with paddle.static.scope_guard(infer_scope):
        # Use fluid.io.load_inference_model to obtain the inference program desc,
        # the feed_target_names (the names of variables that will be feeded
        # data using feed operators), and the fetch_targets (variables that
        # we want to obtain data from using fetch operators).
        logger.info("Constructing the computation graph ...")
        [infer_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(save_dir, infer_exc, 
         model_filename="recognize_digits_%s_test.pdmodel" % DEVICE_SUFFIX, params_filename="recognize_digits_%s.pdiparams" % DEVICE_SUFFIX)
        logger.info("Computation graph built.")


        if FLAGS.use_ipu:
            # TODO(yiakwy) : for the moment, we store our model trained on IPU as static graph
            # which means that the batch size is fixed. 
            # 
            # We will apply passes to trains from batch size to `None` or `-1` upon the generated graph description later
            apply_pseudo_batch_size_pass(infer_program, 1, feed_target_names[0])
        else:
            pass
            # apply_pseudo_batch_size_pass(infer_program, 1, feed_target_names[0])


        if FLAGS.use_ipu:
            # Pipeline with tensorflow frontend: https://docs.graphcore.ai/projects/tensorflow1-user-guide/en/latest/perf_training.html#pipelined-training
            ipu_strategy = compiler.get_ipu_strategy()
            ipu_strategy.is_training = False
            ipu_strategy.num_ipus = num_ipus
            ipu_strategy.enable_pipelining = enable_pipelining
            ipu_compiler = compiler.IpuCompiler(
                infer_program, ipu_strategy=ipu_strategy)
            if will_draw_ir_graph:
                logger.info("Drawing IR graph ...")
                # draw_ir_graph(infer_program, name='infer_graph')
                logger.info("Complete drawing.")
            logger.info("Compiling graph on IPU devices ...")
            feed_list = feed_target_names
            fetch_list = [ out.name for out in fetch_targets]
            infer_program = ipu_compiler.compile(feed_list, fetch_list, infer=True)
            logger.info("Complete compiling.")
        else:
            if will_draw_ir_graph:
                logger.info("Drawing IR graph ...")
                # draw_ir_graph(infer_program, name='infer_graph')
                logger.info("Complete drawing.")


        # infer model
        infer(infer_exc, infer_program, img, feed_target_names, fetch_targets)

    return 0

if __name__ == "__main__":
    sys.exit(main())