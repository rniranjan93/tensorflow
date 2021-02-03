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

using tensorflow::str_util::StartsWith;

namespace xla {
namespace vsiplugin {

Literal BaseVisitor::evaluate(const HloComputation& computation
    /*absl::Span<const Literal* const> arg_literals*/){
    computation.Accept(this);
    return GetEvaluatedLiteralFor(computation.root_instruction()).Clone();
}

se::DeviceMemoryBase BaseVisitor::evaluate(const HloComputation& computation,
    absl::Span<const se::DeviceMemoryBase* const> arg_literals){
    computation.Accept(this);
    auto result_lieral = GetEvaluatedLiteralFor(computation.root_instruction()).Clone();
    
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

            const Literal& lhs_literal = GetEvaluatedLiteralFor(lhs);
            const Literal& rhs_literal = GetEvaluatedLiteralFor(rhs);

            Literal result(shape);
            result.Populate<float>([&](absl::Span<const int64> multi_index) {
                return  lhs_literal.Get<float>(multi_index) +
                        rhs_literal.Get<float>(multi_index);
            });
            evaluated_[hlo] = std::move(result);
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

Status BaseVisitor::HandleConstant(HloInstruction* hlo){
    LOG(INFO) << "PROCESS Constant";
    return Status::OK();
}
}  // namespace poplarplugin
}  // namespace xla
