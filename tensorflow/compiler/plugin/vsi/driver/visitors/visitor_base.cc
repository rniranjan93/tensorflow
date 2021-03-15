/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/plugin/vsi/driver/visitors/visitor_base.h"

#include <stddef.h>

#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tim/vx/operation.h"
#include "tim/vx/ops/elementwise.h"

using tensorflow::str_util::StartsWith;

namespace xla {
namespace vsiplugin {

Literal BaseVisitor::evaluate(const HloComputation& computation
    /*absl::Span<const Literal* const> arg_literals*/){
    computation.Accept(this);
    return GetEvaluatedLiteralFor(computation.root_instruction()).Clone();
}

std::shared_ptr<tim::vx::Tensor> BaseVisitor::evaluate(
    const HloComputation& computation,
    std::vector<Literal>& argument_literals){
    arg_literals_ = std::move(argument_literals);
    computation.Accept(this);
    if (!graph_->Compile()) {
        LOG(FATAL) << "Compile graph fail.";
        return nullptr;
    }
    if(!graph_->Run()){
        LOG(FATAL) << "Run graph fail";
        return nullptr;
    }
    return GetEvaluatedTensorFor(computation.root_instruction());
}

const Shape& BaseVisitor::GetOutputShape(HloInstruction* inst) const {
  return inst->shape();
}

Status BaseVisitor::HandleElementwiseBinary(HloInstruction* hlo){
    switch (hlo->opcode())
    {
        case HloOpcode::kAdd:{
            LOG(INFO) << "PROCESS add";
            auto shape = hlo->shape();
            const HloInstruction* lhs = hlo->operand(0);
            const HloInstruction* rhs = hlo->operand(1);
            TF_RET_CHECK(ShapeUtil::SameDimensions(shape, rhs->shape()));
            TF_RET_CHECK(ShapeUtil::SameDimensions(lhs->shape(), rhs->shape()));

            auto lhs_tensor = GetEvaluatedTensorFor(lhs);
            auto rhs_tensor = GetEvaluatedTensorFor(rhs);
            auto out_tensor = createTensorFromShape(shape, tim::vx::TensorAttribute::OUTPUT);
            auto add = graph_->CreateOperation<tim::vx::ops::Add>();
            (*add).BindInput(lhs_tensor).BindInput(rhs_tensor).BindOutput(out_tensor);

            evaluatedDevMem_[hlo] = executor_->setTensor(out_tensor);
            break;
        }
        default:
            LOG(INFO) << "not has benn implement; opcode:" << hlo->opcode();
            break;
    }
    return Status::OK();
}

Status BaseVisitor::FinishVisit(HloInstruction* root){
    return Status::OK();
}

Status BaseVisitor::Unimplemented(HloInstruction* inst) {
  return xla::Unimplemented("%s (%s) not implemented", inst->name().c_str(),
                            HloOpcodeString(inst->opcode()).c_str());
}

Status BaseVisitor::HandleParameter(HloInstruction* hlo){
    CHECK_LT(hlo->parameter_number(), arg_literals_.size());

    auto& input_literal = arg_literals_[hlo->parameter_number()];
    VLOG(2) << "Parameter evaluated to: " << input_literal.ToString();
    DCHECK(Shape::Equal().MinorToMajorOnlyInLayout()(hlo->shape(),
                                                    input_literal.shape()))
        << "parameter shape is: "
        << ShapeUtil::HumanStringWithLayout(hlo->shape())
        << ", but input literal shape is: "
        << ShapeUtil::HumanStringWithLayout(input_literal.shape());

    if(evaluatedDevMem_.find(hlo) == evaluatedDevMem_.end()){
        ShapeIndex shapeIndex({});
        const float* buffer = input_literal.data<float>(shapeIndex).data();
        auto timTensor = createTensorFromShape(input_literal.shape());
        timTensor->CopyDataToTensor((void *)buffer);
        evaluatedDevMem_[hlo] = executor_->setTensor(timTensor);
    }

    return Status::OK();
}
Status BaseVisitor::HandleConstant(HloInstruction* hlo){
    LOG(INFO) << "PROCESS Constant";
    if(evaluatedDevMem_.find(hlo) == evaluatedDevMem_.end()){
        auto& literal = hlo->literal();

        ShapeIndex shapeIndex({});
        const float* buffer = literal.data<float>(shapeIndex).data();
        auto timTensor = createTensorFromShape(literal.shape());
        timTensor->CopyDataToTensor((void *)buffer);
        evaluatedDevMem_[hlo] = executor_->setTensor(timTensor);
    }

    return Status::OK();
}
}  // namespace poplarplugin
}  // namespace xla
