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

#include "tensorflow/compiler/plugin/vsi/driver/vsi_executable.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/plugin/vsi/driver/vsi_executor.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"

namespace xla{
namespace vsiplugin{

VsiExecutable::VsiExecutable(std::shared_ptr<HloModule> hlo_module) :
        Executable( std::move(hlo_module), 
                    /*hlo_profile_printer_data=*/nullptr,
                    /*hlo_profile_index_map=*/nullptr){}

VsiExecutable::~VsiExecutable()
{
}

StatusOr<ExecutionOutput> VsiExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments,
    HloExecutionProfile* hlo_execution_profile){
        LOG(INFO) << "Execute " << module().name();

        const HloComputation* computation = module().entry_computation();
        LOG(INFO) << "computaion id = " << computation->unique_id();
        LOG(INFO) << "HloComputation->num_parameters' size =  " << computation->num_parameters();
        LOG(INFO) << "HloComputation->parent() = " <<computation->parent();
        LOG(INFO) << "HloComputation->instruction_count() = "<< computation->instruction_count();

        auto root_instr = computation->root_instruction();
        LOG(INFO) << "root instrcution name = " << root_instr->name();
        LOG(INFO) << "root instrcution id = " << root_instr->unique_id();
        LOG(INFO) << "root instrcution operand_count = " << root_instr->operand_count();
        LOG(INFO) << "root instrcution opcode = " << root_instr->opcode();
        for( auto instr : root_instr->operands()){
            LOG(INFO) << "=======================================================";
            LOG(INFO) << "sub instrcution name = " << instr->name();
            LOG(INFO) << "sub instrcution id = " << instr->unique_id();
            LOG(INFO) << "sub instrcution operand_count = " << instr->operand_count();
            LOG(INFO) << "sub instrcution opcode = " << instr->opcode();
            LOG(INFO) << "=======================================================";
        }
        LOG(FATAL)<<"not implement";
    }

StatusOr<std::vector<ScopedShapedBuffer>> VsiExecutable::ExecuteOnStreams(
    absl::Span<const ServiceExecutableRunOptions> run_options,
    absl::Span<const absl::Span<const ShapedBuffer* const>> arguments){
        LOG(FATAL)<<"not implement";
    }

Status VsiExecutable::PopulateExecutionProfile(
    ExecutionProfile* execution_profile,
    HloExecutionProfile* hlo_execution_profile, se::Stream* stream){
        LOG(FATAL)<<"not implement";
    }

} // namespace vsiplugin
} // namespace xla