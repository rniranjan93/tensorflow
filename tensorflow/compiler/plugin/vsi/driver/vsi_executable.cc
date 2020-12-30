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
                    /*hlo_profile_index_map=*/nullptr), 
        visitor_(std::move(std::make_unique<BaseVisitor>())) {
            visitor_->ResetVisitStates();
        }

VsiExecutable::~VsiExecutable()
{
}

StatusOr<ExecutionOutput> VsiExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments,
    HloExecutionProfile* hlo_execution_profile){
        LOG(INFO) << "Execute " << module().name();
        // Convert the ShapeTree to a ShapedBuffer. We do this so we can call
        // TransferManager methods below.
        // std::vector<ShapedBuffer> argument_buffers;
        // argument_buffers.reserve(arguments.size());
        // for (auto& argument : arguments) {
        //     const ShapeTree<MaybeOwningDeviceMemory>& buffers = argument.Buffers();
        //     argument_buffers.push_back(ShapedBuffer(buffers.shape(), buffers.shape(),
        //                                             /*platform=*/nullptr,
        //                                             /*device_ordinal=*/0));
        //     auto in_it = buffers.begin();
        //     auto out_it = argument_buffers.back().buffers().begin();
        //     for (; in_it != buffers.end(); ++in_it, ++out_it) {
        //     out_it->second = in_it->second.AsDeviceMemoryBase();
        //     }
        // }
        const HloComputation* computation = module().entry_computation();
        if (computation->num_parameters() != arguments.size()) {
            return tensorflow::errors::Internal(
            "Mismatch between argument count and graph parameter count.");
        }

        Literal result_literal = visitor_->evaluate(*computation);

        se::Stream* stream = run_options->stream();
        se::StreamExecutor* executor = stream->parent();
        const se::Platform* platform = executor->platform();
        TF_ASSIGN_OR_RETURN(TransferManager * transfer_manager,
                    TransferManager::GetForPlatform(platform));
        // Transform the result literal back into a ShapedBuffer.
        TF_ASSIGN_OR_RETURN(ScopedShapedBuffer result_buffers,
                      transfer_manager->AllocateScopedShapedBuffer(
                          result_literal.shape(), run_options->allocator(),
                          executor->device_ordinal()));

        TF_RETURN_IF_ERROR(transfer_manager->TransferLiteralToDevice(
        run_options->stream(), result_literal, result_buffers));
        ExecutionOutput result(std::move(result_buffers));
        return result;
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