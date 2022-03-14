#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

# Run:
# export OMPI_ALLOW_RUN_AS_ROOT=1
# export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
# poprun --host=localhost --num-instances 2 --num-replicas 4  --ipus-per-replica 1 --print-topology=yes python3.7 test_distribution_ipu.py

import numpy as np
import paddle

comm = None


def TestDist():
    paddle.enable_static()

    attrs = {"size": [128, 16], "padding_idx": -1, "dtype": 'float32'}

    scope = paddle.fluid.core.Scope()
    main_prog = paddle.static.Program()
    startup_prog = paddle.static.Program()
    main_prog.random_seed = 42
    startup_prog.random_seed = 42

    with paddle.fluid.scope_guard(scope):
        with paddle.static.program_guard(main_prog, startup_prog):
            x = paddle.static.data(name="x", shape=[3, 2, 1], dtype='int64')

            out = paddle.fluid.layers.embedding(x, **attrs)
            out = paddle.mean(out)

            feed_list = ["x"]
            fetch_list = [out.name]

            place = paddle.IPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(startup_prog)

            ipu_strategy = paddle.static.IpuStrategy()
            ipu_strategy.set_graph_config(num_ipus=2, is_training=False)
            # Enable the distributed computing
            ipu_strategy.set_options({
                "enable_distribution": True,
                "enable_replicated_graphs": True,
                "replicated_graph_count": 2,
                "global_replication_factor": 4
            })

            program = paddle.static.IpuCompiledProgram(
                main_prog,
                ipu_strategy=ipu_strategy).compile(feed_list, fetch_list)

            input_data = np.concatenate([
                np.array(
                    [[[1], [3]], [[2], [4]], [[4], [127]]]).astype(np.int32),
                np.array(
                    [[[1], [3]], [[2], [4]], [[4], [127]]]).astype(np.int32)
            ])
            feed_data = {"x": input_data}

            res = exe.run(program, feed=feed_data, fetch_list=fetch_list)
            res = comm.reduce(res, root=0)
            if rank == 0:
                res = np.mean(np.concatenate(res))
                print(res)


if __name__ == "__main__":
    from mpi4py import MPI

    DISTRIBUTED_COMM = MPI.COMM_WORLD

    def _get_comm():
        global DISTRIBUTED_COMM
        if DISTRIBUTED_COMM is None:
            raise RuntimeError(
                "Distributed Commumication not setup. Please run setup_comm(MPI.COMM_WORLD) first. "
                "See https://mpi4py.readthedocs.io/ for details on MPI.COMM_WORLD."
            )
        return DISTRIBUTED_COMM

    comm = _get_comm()
    size = comm.Get_size()
    rank = comm.Get_rank()

    TestDist()
