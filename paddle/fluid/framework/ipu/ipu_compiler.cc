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

#include "paddle/fluid/framework/ipu/ipu_compiler.h"

#include "paddle/fluid/framework/ipu/ipu_utils.h"
#include "paddle/fluid/framework/ir/graph_helper.h"

namespace paddle {
namespace framework {
namespace ipu {

template <typename T>
T GetAttrAllowNull(std::string attr, OpDesc* op_desc) {
  if (op_desc->HasAttr(attr)) {
    return BOOST_GET_CONST(T, op_desc->GetAttr(attr));
  } else {
    return {};
  }
}

template <typename T>
nonstd::optional<T> GetOptAttrAllowNull(std::string attr, OpDesc* op_desc) {
  if (op_desc->HasAttr(attr)) {
    return BOOST_GET_CONST(T, op_desc->GetAttr(attr));
  } else {
    return {};
  }
}

template <typename TI, typename TO>
TO GetCastSigAttrAllowNull(std::string attr, OpDesc* op_desc) {
  if (op_desc->HasAttr(attr)) {
    auto x = BOOST_GET_CONST(TI, op_desc->GetAttr(attr));
    return static_cast<TO>(x);
  } else {
    return {};
  }
}

Compiler::Compiler() {
  builder_ = popart::Builder::create();
  RegisterOpFunc();
}

Compiler::~Compiler() {}

void Compiler::RegisterOpFunc() {
  VLOG(10) << "enter Compiler::RegisterOpFunc";
#define INT_VEC std::vector<std::int64_t>
#define INT32_VEC std::vector<std::int32_t>
#define FLOAT_VEC std::vector<float>
#define FLOAT float
#define INT std::int64_t
#define INT32 std::int32_t
#define BOOL bool
#define STRING std::string
#define STRING_VEC std::vector<std::string*>
#define NONE

#define ARG(Type, Name) , GetAttrAllowNull<Type>(#Name, op_desc)
#define OPT_ARG(Type, Name) , GetOptAttrAllowNull<Type>(#Name, op_desc)
#define SIG_ARG(TI, TO, Name) , GetCastSigAttrAllowNull<TI, TO>(#Name, op_desc)
#define POPART_CONST_ARG(Name) , const PopartConstant& Name
#define HOST_SIDE_CONST_ARG(Name) , const HostSideConstant& Name
#define POPART_ATTRIB_VEC_ARG(Name)
#define BODY_ARG(Name) NONE

  name_function_ = {
#define OP_DECL(FuncName, OnnxImpl, Args)                     \
  {#FuncName, [&](OpDesc* op_desc) {                          \
     auto op_type = op_desc->Type();                          \
     VLOG(10) << "build op:" << op_type << " args " << #Args; \
     auto inputs = GetOpInputs(op_desc);                      \
     auto output_names = GetOpOutputs(op_desc);               \
     auto debug_context = BuildDebugContext(op_desc);         \
     auto aiGraphcoreOpset = builder_->aiGraphcoreOpset1();   \
     auto aiOnnxOpset = builder_->aiOnnxOpset11();            \
     auto output_ids = OnnxImpl(inputs Args, debug_context);  \
     SetIpuIndexStage(output_ids, op_desc);                   \
     SetAMPAttributes(output_ids, op_desc);                   \
     SetSerializeAttributes(output_ids, op_desc);             \
     InsertTensors(output_names, output_ids);                 \
   }},  // NOLINT
#include "paddle/fluid/framework/ipu/supported_ops.h"
  };

#undef OP_DECL
#undef BODY_ARG
#undef POPART_ATTRIB_VEC_ARG
#undef HOST_SIDE_CONST_ARG
#undef POPART_CONST_ARG
#undef SIG_ARG
#undef OPT_ARG
#undef ARG
#undef NONE
#undef STRING_VEC
#undef STRING
#undef BOOL
#undef INT32
#undef INT
#undef FLOAT
#undef FLOAT_VEC
#undef INT32_VEC
#undef INT_VEC
}

void Compiler::LowerBody(const ir::Graph* graph) {
  VLOG(10) << "enter Compiler::LowerBody";
  auto nodes = ir::TopologySortOperations(*graph);
  for (auto* node : nodes) {
    auto* op_desc = node->Op();
    auto op_type = op_desc->Type();
    VLOG(10) << "node->type: " << op_type;

    if (op_type == "popart_constant") {
      auto dims =
          BOOST_GET_CONST(std::vector<int64_t>, op_desc->GetAttr("dims"));
      auto dtype_ = BOOST_GET_CONST(int, op_desc->GetAttr("dtype"));
      auto dtype = OnnxDtype2PopartType(dtype_);
      popart::TensorInfo tensor_info{dtype, dims};
      auto value_attr = op_desc->GetAttr("value");
      auto const_data = std::unique_ptr<popart::ConstVoidData>{};

      auto convert2fp16 = [&](std::vector<float> data) -> void* {
        int data_num = data.size();
        uint16_t* fp16_malloc = new uint16_t[data_num];
        std::vector<uint16_t> fp16_data;
        std::transform(data.data(), data.data() + data_num,
                       std::back_inserter(fp16_data),
                       [&](float elem) { return popart::floatToHalf(elem); });
        memcpy(reinterpret_cast<void*>(fp16_malloc), fp16_data.data(),
               data_num * sizeof(uint16_t));
        return fp16_malloc;
      };
      switch (dtype) {
        case popart::DataType::FLOAT16:
          const_data.reset(new popart::ConstVoidData(
              convert2fp16(BOOST_GET_CONST(std::vector<float>, value_attr)),
              tensor_info));
          break;
        case popart::DataType::FLOAT:
          const_data.reset(
              new popart::ConstVoidData(DynamicMalloc<float>(BOOST_GET_CONST(
                                            std::vector<float>, value_attr)),
                                        tensor_info));
          break;
        case popart::DataType::INT32:
          const_data.reset(new popart::ConstVoidData(
              DynamicMalloc<int>(BOOST_GET_CONST(std::vector<int>, value_attr)),
              tensor_info));
          break;
        case popart::DataType::DOUBLE:
          const_data.reset(
              new popart::ConstVoidData(DynamicMalloc<double>(BOOST_GET_CONST(
                                            std::vector<double>, value_attr)),
                                        tensor_info));
          break;
        case popart::DataType::INT64:
          const_data.reset(
              new popart::ConstVoidData(DynamicMalloc<int64_t>(BOOST_GET_CONST(
                                            std::vector<int64_t>, value_attr)),
                                        tensor_info));
          break;
        default:
          PADDLE_THROW(
              platform::errors::Unimplemented("popart::DataType %d", dtype));
      }

      popart::TensorId result = builder_->aiOnnxOpset11().constant(*const_data);
      SetIpuIndexStage(result, op_desc);
      InsertTensors(GetOpOutputs(op_desc), result);
    } else if (op_type == "popart_custom_op") {
      auto inputs = GetOpInputs(op_desc);
      auto outputs = GetOpOutputs(op_desc);
      auto debug_context = BuildDebugContext(op_desc);
      auto attributes = std::map<std::string, popart::any>{};
      for (auto& attr : op_desc->GetAttrMap()) {
        CustomOpAttrVisitor visitor(&attributes, attr.first);
        boost::apply_visitor(visitor, attr.second);
      }
      auto __op_type =
          BOOST_GET_CONST(std::string, op_desc->GetAttr("__op_type"));
      VLOG(10) << "Build graph from custom op: " << __op_type;
      auto it = custom_ops_.find(__op_type);
      auto output_ids =
          builder_->customOp(it->second.popart_op, it->second.popart_op.version,
                             inputs, outputs.size(), attributes, debug_context);
      SetIpuIndexStage(output_ids, op_desc);
      InsertTensors(outputs, output_ids);
    } else if (op_type == "popart_printtensor") {
      auto inputs = GetOpInputs(op_desc);
      auto outputs = GetOpOutputs(op_desc);
      auto debug_context = BuildDebugContext(op_desc);
      auto print_gradient =
          BOOST_GET_CONST(int64_t, op_desc->GetAttr("print_gradient"));
      auto title = BOOST_GET_CONST(std::string, op_desc->GetAttr("title"));
      auto output_ids = builder_->aiGraphcoreOpset1().printtensor(
          inputs, print_gradient, debug_context, title);
      SetIpuIndexStage(output_ids, op_desc);
      InsertTensors(outputs, output_ids);
    } else {
      auto itr = name_function_.find(op_type);
      if (itr != name_function_.end()) {
        itr->second(node->Op());
      } else {
        PADDLE_THROW(
            platform::errors::NotFound("%s is not registered", op_type));
      }
    }
  }
  VLOG(10) << "leave Compiler::LowerBody";
}

void Compiler::InitInputs(ir::Graph* graph,
                          const std::vector<std::string>& feed_list) {
  for (const auto& feed_name : feed_list) {
    feed_list_.push_back(feed_name);
    for (const ir::Node* n : graph->Nodes()) {
      if (n->IsVar()) {
        auto* var_desc = n->Var();
        if (feed_name == var_desc->Name()) {
          VLOG(10) << "feed_name= " << var_desc->Name();
          auto data_type = VarType2PopartType(var_desc->GetDataType());
          popart::TensorInfo input_info{data_type, var_desc->GetShape()};
          VLOG(10) << "popart input_info = " << input_info;
          popart::TensorId tensor_id =
              builder_->addInputTensor(input_info, feed_name);
          VLOG(10) << "popart input tensor id = " << tensor_id;
          inputs_.push_back(tensor_id);
          tensors_.emplace(var_desc->Name(), tensor_id);
        }
      }
    }
  }
}

void Compiler::InitOutputs(const std::vector<std::string>& fetch_list) {
  for (const auto& fetch_name : fetch_list) {
    fetch_list_.push_back(fetch_name);
    auto tensor = tensors_.find(fetch_name);
    PADDLE_ENFORCE_NE(tensor, tensors_.end(),
                      platform::errors::NotFound(
                          "output tensor %s does not exist.", fetch_name));
    VLOG(10) << "fetch_name= " << fetch_name;
    VLOG(10) << "popart output tensor id = " << tensor->second;
    builder_->addOutputTensor(tensor->second);
    outputs_.push_back(tensor->second);
  }
}

void Compiler::LowerWeights(const ir::Graph* graph, const Scope* scope_) {
  PADDLE_ENFORCE_NOT_NULL(scope_,
                          platform::errors::PreconditionNotMet(
                              "You should call set_scope before LowerWeights"));
  // at this step, the graph doesn't contains optimizer related states
  for (const auto* node : graph->Nodes()) {
    if (node->IsVar() && !node->IsCtrlVar() && node->Var()) {
      if (node->Var()->Persistable() && node->inputs.empty()) {
        auto var_name = node->Var()->Name();
        // workround: https://github.com/graphcore/Paddle/issues/151
        if (tensors_.count(var_name) != 0) {
          continue;
        }

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
              builder_->addInitializedInputTensor(const_data, var_name);
          tensors_.emplace(var_name, result);
          weights_.push_back(result);
        }
      }
    }
  }
}

void Compiler::InsertTensors(const std::vector<std::string>& output_names,
                             const std::vector<std::string>& tensor_ids) {
  PADDLE_ENFORCE_EQ(output_names.size(), tensor_ids.size(),
                    platform::errors::Fatal("InsertTensors size mismatch"));
  for (int i = 0; i < tensor_ids.size(); i++) {
    std::string tensor_id = tensor_ids[i];
    tensors_.emplace(output_names[i], tensor_ids[i]);
  }
}

void Compiler::InsertTensors(const std::vector<std::string>& output_names,
                             const std::string& tensor_id) {
  PADDLE_ENFORCE_EQ(output_names.size(), 1,
                    platform::errors::Fatal("InsertTensors size mismatch"));
  tensors_.emplace(output_names[0], tensor_id);
}

void Compiler::SetIpuIndexStage(const std::vector<std::string>& tensor_ids,
                                const OpDesc* op_desc) {
  VLOG(10) << "enter Compiler::SetIpuIndexStage";
  auto tensor_ids_set =
      std::set<std::string>(tensor_ids.begin(), tensor_ids.end());

  if (op_desc->HasAttr(sIpuIndexAttr)) {
    auto ipu_index = BOOST_GET_CONST(int, op_desc->GetAttr(sIpuIndexAttr));
    builder_->virtualGraph(tensor_ids_set, ipu_index);
    VLOG(10) << "set " << sIpuIndexAttr << " = " << ipu_index
             << " for op: " << op_desc->Type();
    if (op_desc->HasAttr(sIpuStageAttr)) {
      auto ipu_stage = BOOST_GET_CONST(int, op_desc->GetAttr(sIpuStageAttr));
      builder_->pipelineStage(tensor_ids_set, ipu_stage);
      VLOG(10) << "set " << sIpuStageAttr << "= " << ipu_stage
               << " for op: " << op_desc->Type();
    }
  }
  VLOG(10) << "leave Compiler::SetIpuIndexStage";
}

void Compiler::SetIpuIndexStage(const std::string& tensor_id,
                                const OpDesc* op_desc) {
  VLOG(10) << "enter Compiler::SetIpuIndexStage";

  if (op_desc->HasAttr(sIpuIndexAttr)) {
    auto ipu_index = BOOST_GET_CONST(int, op_desc->GetAttr(sIpuIndexAttr));
    builder_->virtualGraph(tensor_id, ipu_index);
    VLOG(10) << "set " << sIpuIndexAttr << " = " << ipu_index
             << " for op: " << op_desc->Type();
    if (op_desc->HasAttr(sIpuStageAttr)) {
      auto ipu_stage = BOOST_GET_CONST(int, op_desc->GetAttr(sIpuStageAttr));
      builder_->pipelineStage(tensor_id, ipu_stage);
      VLOG(10) << "set " << sIpuStageAttr << "= " << ipu_stage
               << " for op: " << op_desc->Type();
    }
  }
  VLOG(10) << "leave Compiler::SetIpuIndexStage";
}

void Compiler::SetAMPAttributes(const std::vector<std::string>& tensor_ids,
                                const OpDesc* op_desc) {
  if (op_desc->Type() == "popart_matmul") {
    for (const auto& tensor_id : tensor_ids) {
      SetAMPAttributes(tensor_id, op_desc);
    }
  }
}

void Compiler::SetAMPAttributes(const std::string& tensor_id,
                                const OpDesc* op_desc) {
  VLOG(10) << "enter Compiler::SetAMPAttributes";
  if (op_desc->Type() == "popart_matmul") {
    auto amp = ipu_strategy_->available_memory_proportion;
    if (amp > 0.0f && amp <= 1.0) {
      builder_->setAvailableMemoryProportion(tensor_id, amp);
    }
  }
  VLOG(10) << "leave Compiler::SetAMPAttributes";
}

void Compiler::SetSerializeAttributes(
    const std::vector<std::string>& tensor_ids, const OpDesc* op_desc) {
  VLOG(10) << "enter Compiler::SetSerializeAttributes";
  auto tensor_ids_set =
      std::set<std::string>(tensor_ids.begin(), tensor_ids.end());

  if (op_desc->Type() == "popart_matmul") {
    if (op_desc->HasAttr(sMatmulSerializeFactor)) {
      auto factor =
          BOOST_GET_CONST(int, op_desc->GetAttr(sMatmulSerializeFactor));
      std::string mode = "output_channels";
      if (op_desc->HasAttr(sMatmulSerializeMode)) {
        mode = BOOST_GET_CONST(std::string,
                               op_desc->GetAttr(sMatmulSerializeMode));
      }
      builder_->setSerializeMatMul(tensor_ids_set, mode, (int64_t)factor, true);
    }
  }
  VLOG(10) << "leave Compiler::SetSerializeAttributes";
}

void Compiler::SetSerializeAttributes(const std::string& tensor_id,
                                      const OpDesc* op_desc) {
  std::vector<std::string> tensor_ids = {tensor_id};
  SetSerializeAttributes(tensor_ids, op_desc);
}

void Compiler::SetIpuStrategy(const IpuStrategy& strategy) {
  ipu_strategy_ = &strategy;
}

void Compiler::SetCustomOps(
    const std::vector<IpuCustomOpIdentifier>& custom_ops) {
  for (auto x : custom_ops) {
    custom_ops_.emplace(x.paddle_op, x);
  }
}

std::vector<popart::TensorId>& Compiler::GetWeights() { return weights_; }

// convertFloatsToHalfs
void Compiler::ConvertProtoToFp16() {
  popart::GraphTransformer graph_transformer(builder_->getModelProto());
  graph_transformer.convertFloatsToHalfs();
  converted_proto_ = graph_transformer.getModelProto();
}

std::string Compiler::GetModelProto() {
  if (converted_proto_.length()) {
    return converted_proto_;
  }
  return builder_->getModelProto();
}

void Compiler::SaveModelProto(const std::string& path) {
  builder_->saveModelProto(path);
}

void Compiler::SaveModelProtoNoCheck(const std::string& path) {
  auto proto = GetModelProto();
  std::ofstream onnxfile(path, std::ios_base::binary);
  onnxfile.write(proto.data(), proto.size());
  onnxfile.close();
}

std::vector<std::string> Compiler::GetOpInputs(const OpDesc* op) {
  auto ins = op->Input("__inputs__");
  std::vector<std::string> inputs;
  for (const auto& in : ins) {
    if (tensors_.find(in) != tensors_.end()) {
      inputs.push_back(tensors_[in]);
    } else {
      inputs.push_back(in);
    }
  }
  return inputs;
}

const std::vector<std::string>& Compiler::GetOpOutputs(const OpDesc* op) {
  return op->Output("__outputs__");
}

popart::DebugContext Compiler::BuildDebugContext(const OpDesc* op) {
  auto op_identify_id =
      BOOST_GET_CONST(std::string, op->GetAttr(sOpIdentifyIdAttr));
  VLOG(10) << "op_identify_id of op: " << op->Type() << " is "
           << op_identify_id;
  return popart::DebugContext(op_identify_id);
}

}  // namespace ipu
}  // namespace framework
}  // namespace paddle
