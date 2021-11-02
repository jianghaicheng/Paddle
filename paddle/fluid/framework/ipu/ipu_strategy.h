/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <popart/op.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/tensorlocation.hpp>

namespace paddle {
namespace framework {
namespace ipu {

using VirtualGraphMode = popart::VirtualGraphMode;
using RecomputationType = popart::RecomputationType;

struct IpuStrategy {
  IpuStrategy() {
    // we always save optimizer state to OffChip and enable rts for saving
    // memory
    auto storage = popart::TensorLocation(popart::TensorStorage::OffChip,
                                          popart::ReplicatedTensorSharding::On);
    popart_options.optimizerStateTensorLocationSettings =
        popart::TensorLocationSettings(storage);

    // We divide the accumulationFactor and replicatedGraphCount after all
    // reduce
    popart_options.accumulationAndReplicationReductionType =
        popart::ReductionType::Mean;
    popart_options.meanAccumulationAndReplicationReductionStrategy =
        popart::MeanReductionStrategy::Post;
  }
  ~IpuStrategy() {}

  // Number ipus total needed, replica * ipu_per_replica
  int num_ipus = 1;

  // batches per step
  int batches_per_step = 1;

  // micro batch-size
  int batch_size = 1;

  // training flag, true for training
  bool is_training = true;

  // save the onnx model lowered by paddle program description
  bool save_init_onnx = false;

  // save the trained model
  bool save_last_onnx = false;

  // save paddle model per n steps
  int save_per_n_step = 1;

  // average sharding, debugging used
  bool need_avg_shard = false;

  // flag for fp16, true for pure fp16
  bool enable_fp16 = false;

  // available memory proportion, 0.0f for disable
  float available_memory_proportion = 0.0f;

  // popart session option
  popart::SessionOptions popart_options;
};

}  // namespace ipu
}  // namespace framework
}  // namespace paddle
