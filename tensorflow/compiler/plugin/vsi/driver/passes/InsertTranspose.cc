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

#include "tensorflow/compiler/plugin/vsi/driver/passes/InsertTranspose.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"

namespace xla {
namespace vsiplugin {

StatusOr<bool> InsertTranspose::Run(HloModule* module){
    bool changed = false;
    XLA_VLOG_LINES(2, "InsertTranspose::Run(), before:\n" + module->ToString());

    for (HloComputation* computation : module->MakeComputationPostOrder()) {
        for (HloInstruction* instruction : computation->MakeInstructionPostOrder()) {
            if (instruction->opcode() == HloOpcode::kConvolution) {
                changed |= InsertTransposeForConv(instruction).ValueOrDie();
            }
        }
    }
    XLA_VLOG_LINES(2,
            "InsertTranspose::Run(), after:\n" + module->ToString());
    return changed;
}

StatusOr<bool> InsertTranspose::InsertTransposeForConv(HloInstruction *conv){
    auto lhs = conv->operand(0);
    auto rhs = conv->operand(1);
    const auto& window = conv->window();
    const Shape& result_shape = conv->shape();
    const Shape& lhs_shape = lhs->shape();
    const Shape& rhs_shape = rhs->shape();

    TF_CHECK_OK(ShapeUtil::ValidateShape(lhs_shape));
    TF_CHECK_OK(ShapeUtil::ValidateShape(rhs_shape));
    CHECK(lhs_shape.IsArray());
    CHECK(rhs_shape.IsArray());
    CHECK(ShapeUtil::SameElementType(lhs_shape, rhs_shape));
    CHECK(ShapeUtil::SameElementType(lhs_shape, result_shape));

    const auto& dnums = conv->convolution_dimension_numbers();
    const int64 num_spatial_dims = dnums.output_spatial_dimensions_size();
    CHECK_EQ(num_spatial_dims, dnums.input_spatial_dimensions_size());
    CHECK_EQ(num_spatial_dims, dnums.kernel_spatial_dimensions_size());
    CHECK_EQ(num_spatial_dims, 2); /*vsi requirement*/
    CHECK_GE(num_spatial_dims, 0);
    CHECK_EQ(window.dimensions_size(), num_spatial_dims);

    const auto lhs_rank = lhs_shape.rank();
    const auto rhs_rank = rhs_shape.rank();
    CHECK_EQ(num_spatial_dims + 2, lhs_rank);
    CHECK_EQ(num_spatial_dims + 2, rhs_rank);

    TF_ASSIGN_OR_RETURN(auto inferred_return_shape,
                        ShapeInference::InferConvolveShape(
                            lhs_shape, rhs_shape, conv->feature_group_count(),
                            conv->batch_group_count(), window, dnums));
    CHECK(ShapeUtil::Compatible(result_shape, inferred_return_shape))
        << "return shape set to: " << ShapeUtil::HumanString(result_shape)
        << " but is inferred to be: "
        << ShapeUtil::HumanString(inferred_return_shape);

    std::vector<uint32_t> input_dim;
    input_dim.push_back(dnums.input_batch_dimension());
    input_dim.push_back(dnums.input_feature_dimension());

    std::vector<uint32_t> weight_dim;
    weight_dim.push_back(dnums.kernel_output_feature_dimension());
    weight_dim.push_back(dnums.kernel_input_feature_dimension());
    for(size_t i = 2; i < lhs_rank; i++){
        input_dim.push_back(dnums.input_spatial_dimensions(i - 2));
        weight_dim.push_back(dnums.kernel_spatial_dimensions(i - 2));
    }

}
}  // namespace vsiplugin
}  // namespace xla
