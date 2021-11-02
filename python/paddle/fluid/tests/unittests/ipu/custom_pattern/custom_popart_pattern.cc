// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <popart/op/relu.hpp>
#include <popart/patterns/patterns.hpp>

class PrintRelu : public popart::PreAliasPattern {
 public:
  bool matches(popart::Op *op) const override {
    return op->isConvertibleTo<popart::ReluOp>();
  }

  std::vector<const popart::Tensor *> touches(popart::Op *) const override {
    return {};
  }

  bool apply(popart::Op *op) const override {
    // just print every relu op in th graph
    // this pattern will be called multiple times during compiling

    popart::logging::warn("call PrintOpPattern::apply");
    popart::logging::warn("{}", op->debugName());

    return false;
  }
};

static popart::PatternCreator<PrintRelu> PrintReluPatternCreator(
    "PrintReluPatternPattern", true);
