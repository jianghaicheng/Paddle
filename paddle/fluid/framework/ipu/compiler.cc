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
#include "paddle/fluid/framework/ipu/compiler.h"

namespace paddle {
namespace framework {
namespace ipu {

Compiler::Compiler(const IpuStrategy* ipu_strategy) {
  ipu_strategy_ = ipu_strategy;
  builder_ = popart::Builder::create();
  RegisterOpFunc();
}

Compiler::~Compiler() {
  builder_.release();
  tensors_.clear();
  ipu_strategy_ = nullptr;
}

void Compiler::InsertTensors(std::vector<std::string> output_names,
                             std::vector<std::string> tensor_ids) {
  for (int i = 0; i < tensor_ids.size(); i++) {
    std::string tensor_id = tensor_ids[i];
    tensors_.emplace(output_names[i], tensor_ids[i]);
  }
}

void Compiler::InsertTensors(std::vector<std::string> output_names,
                             std::string tensor_id) {
  tensors_.insert(
      std::pair<std::string, std::string>(output_names[0], tensor_id));
}

void Compiler::SetIpuIndexStage(const std::vector<std::string>& tensor_ids,
                                const OpDesc* op_desc) {
  // TODO(xiaobingw): replace ipu_index with macro or constexpr
  VLOG(10) << "enter Compiler::SetIpuIndexStage";
  auto tensor_ids_set =
      std::set<std::string>(tensor_ids.begin(), tensor_ids.end());
  if (op_desc->HasAttr("ipu_index")) {
    auto ipu_index = BOOST_GET_CONST(int, op_desc->GetAttr("ipu_index"));
    builder_->virtualGraph(tensor_ids_set, ipu_index);
    VLOG(10) << "set ipu_index= " << ipu_index
             << " for op: " << op_desc->Type();
    if (op_desc->HasAttr("ipu_stage")) {
      auto ipu_stage = BOOST_GET_CONST(int, op_desc->GetAttr("ipu_stage"));
      builder_->pipelineStage(tensor_ids_set, ipu_stage);
      VLOG(10) << "set ipu_stage= " << ipu_stage
               << " for op: " << op_desc->Type();
    }
  }
  VLOG(10) << "leave Compiler::SetIpuIndexStage";
}

void Compiler::SetIpuIndexStage(const std::string& tensor_id,
                                const OpDesc* op_desc) {
  VLOG(10) << "enter Compiler::SetIpuIndexStage";
  if (op_desc->HasAttr("ipu_index")) {
    auto ipu_index = BOOST_GET_CONST(int, op_desc->GetAttr("ipu_index"));
    builder_->virtualGraph(tensor_id, ipu_index);
    VLOG(10) << "set ipu_index= " << ipu_index
             << " for op: " << op_desc->Type();
    if (op_desc->HasAttr("ipu_stage")) {
      auto ipu_stage = BOOST_GET_CONST(int, op_desc->GetAttr("ipu_stage"));
      builder_->pipelineStage(tensor_id, ipu_stage);
      VLOG(10) << "set ipu_stage= " << ipu_stage
               << " for op: " << op_desc->Type();
    }
  }
  VLOG(10) << "leave Compiler::SetIpuIndexStage";
}

template <typename T>
T GetAttrAllowNull(std::string attr, OpDesc* op_desc) {
  std::string type = typeid(T).name();
  VLOG(1) << "body attr type is: " << type << " body attr name is: " << attr;
  if (op_desc->HasAttr(attr)) {
    return BOOST_GET_CONST(T, op_desc->GetAttr(attr));
  } else {
    VLOG(1) << "body attr not exist: " << type;
    return {};
  }
}

void Compiler::InitInputs(ir::Graph* graph,
                          const std::vector<std::string>& feed_list) {
  for (const auto& feed_name : feed_list) {
    VLOG(1) << feed_name;

    for (const ir::Node* n : graph->Nodes()) {
      if (n->IsVar()) {
        auto* var_desc = n->Var();
        if (feed_name == var_desc->Name()) {
          // Get tensor_info from var_desc
          VLOG(1) << "feed_name= " << var_desc->Name();
          auto data_type = VarType2PopartType(var_desc->GetDataType());
          popart::TensorInfo input_info{data_type, var_desc->GetShape()};
          // Create popart tensor
          VLOG(1) << "popart input_info = " << input_info;
          popart::TensorId tensor_id = builder_->addInputTensor(input_info);
          VLOG(1) << "popart input tensor id = " << tensor_id;
          inputs_.push_back(tensor_id);
          tensors_.emplace(var_desc->Name(), tensor_id);
        }
      }
    }
  }
}

void Compiler::InitOutputs(const std::vector<std::string>& fetch_list) {
  for (const auto& fetch_name : fetch_list) {
    auto tensor = tensors_.find(fetch_name);
    PADDLE_ENFORCE_NE(tensor, tensors_.end(),
                      platform::errors::NotFound(
                          "output tensor %s does not exist.", fetch_name));
    VLOG(1) << "fetch_name= " << fetch_name;
    VLOG(1) << "popart output tensor id = " << tensor->second;
    builder_->addOutputTensor(tensor->second);
    outputs_.push_back(tensor->second);
  }
}

void Compiler::LowerWeights(const ir::Graph* graph, const Scope* scope_) {
  PADDLE_ENFORCE_NOT_NULL(scope_,
                          platform::errors::PreconditionNotMet(
                              "You should call set_scope before LowerWeights"));
  // at this step, i think the graph doesn't contains optimizer
  // related states
  for (const auto* node : graph->Nodes()) {
    if (node->IsVar() && !node->IsCtrlVar() && node->Var()) {
      if (node->Var()->Persistable()) {
        auto var_name = node->Var()->Name();
        auto var = scope_->FindVar(var_name);
        if (var) {
          auto tensor = var->Get<framework::LoDTensor>();
          auto dtype = VarType2PopartType(tensor.type());
          auto shape = std::vector<int64_t>();
          for (size_t i = 0; i < tensor.dims().size(); ++i) {
            shape.push_back(tensor.dims().at(i));
          }
          popart::TensorInfo tensor_info(dtype, shape);
          popart::ConstVoidData const_data{tensor.data<void>(), tensor_info};
          popart::TensorId result =
              builder_->addInitializedInputTensor(const_data);
          tensors_.emplace(var_name, result);
        }
      }
    }
  }
}

std::vector<std::string> Compiler::GetOpInputs(const OpDesc* op) {
  auto inputs_ = op->Input("__inputs__");
  std::vector<std::string> inputs;
  for (const auto& in : inputs_) {
    if (tensors_.find(in) != tensors_.end()) {
      inputs.push_back(tensors_[in]);
    } else {
      inputs.push_back(in);
    }
  }
  return inputs;
}

void Compiler::RegisterOpFunc() {
  VLOG(1) << "enter Compiler::RegisterOpFunc";
#define INT_VEC std::vector<std::int64_t>
#define FLOAT_VEC std::vector<float>
#define FLOAT float
#define INT std::int64_t
#define BOOL bool
#define STRING std::string
#define STRING_VEC std::vector<std::string*>
#define NONE

#define ARG(Type, Name) , GetAttrAllowNull<Type>(#Name, op_desc)
#define POPART_CONST_ARG(Name) , const PopartConstant& Name
#define HOST_SIDE_CONST_ARG(Name) , const HostSideConstant& Name
#define POPART_ATTRIB_VEC_ARG(Name)
#define BODY_ARG(Name) NONE

  name_function_ = {
#define OP_DECL(FuncName, OnnxImpl, Args)                    \
  {#FuncName, [&](OpDesc* op_desc) {                         \
     auto op_type = op_desc->Type();                         \
     VLOG(1) << "build op:" << op_type << " args " << #Args; \
     auto inputs = GetOpInputs(op_desc);                     \
     auto output_names = op_desc->Output("__outputs__");     \
     auto aiOnnxOpset1 = builder_->aiGraphcoreOpset1();      \
     auto aiOnnxOpset = builder_->aiOnnxOpset11();           \
     auto output_ids = OnnxImpl(inputs Args);                \
     SetIpuIndexStage(output_ids, op_desc);                  \
     InsertTensors(output_names, output_ids);                \
   }},
#include "paddle/fluid/framework/ipu/supported_ops_autogen.h"
  };

#undef OP_DECL
// #undef OP_DECL_NO_RETURN
#undef BODY_ARG
#undef POPART_ATTRIB_VEC_ARG
#undef HOST_SIDE_CONST_ARG
#undef POPART_CONST_ARG
#undef ARG
#undef NONE
#undef STRING_VEC
#undef STRING
#undef BOOL
#undef INT
#undef FLOAT
#undef FLOAT_VEC
#undef INT_VEC

// self register ops
#include "paddle/fluid/framework/ipu/supported_ops_custom.h"
  name_function_.emplace("popart_reducemean", ReduceMeanHandler);
  name_function_.emplace("popart_batchnormalization", BatchNormHandler);
  name_function_.emplace("popart_constant", Constant);
  name_function_.emplace("popart_nllloss", NllLoss);
  name_function_.emplace("popart_groupnormalization", Groupnormalization);
}

void Compiler::LowerBody(const ir::Graph* graph) {
  VLOG(10) << "enter Compiler::LowerBody";
  // used for debug
  for (auto elem : name_function_) {
    VLOG(1) << "registered in map : " << elem.first << " second "
            << &(elem.second);
  }
  auto nodes = paddle::framework::ir::TopologySortOperations(*graph);
  for (auto* node : nodes) {
    OpDesc* op = node->Op();
    VLOG(1) << "node->type: " << op->Type();
    PADDLE_ENFORCE_GT(
        name_function_.count(op->Type()), 0,
        platform::errors::NotFound(
            "Do not found operator convert function, please make "
            "sure it is registered in file \"supported_ops_autogen.h\" or "
            "\"supported_ops_custom.h\""));
    auto func = name_function_[op->Type()];
    func(node->Op());
  }
  VLOG(10) << "leave Compiler::LowerBody";
}

}  // namespace ipu
}  // namespace framework
}  // namespace paddle
