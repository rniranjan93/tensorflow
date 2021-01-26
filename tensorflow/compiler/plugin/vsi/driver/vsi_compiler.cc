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

#include "tensorflow/compiler/plugin/vsi/driver/vsi_compiler.h"

#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/cholesky_expander.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/dynamic_index_splitter.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"
#include "tensorflow/compiler/xla/service/layout_assignment.h"
#include "tensorflow/compiler/xla/service/map_inliner.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/service/triangular_solve_expander.h"
#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"

#include "tensorflow/compiler/plugin/vsi/driver/vsi_platform_id.h"
#include "tensorflow/compiler/plugin/vsi/driver/vsi_executable.h"

namespace xla {
    namespace vsiplugin {

    StatusOr<std::unique_ptr<HloModule>> VsiCompiler::RunHloPasses(
        std::unique_ptr<HloModule> hlo_module, se::StreamExecutor* stream_exec,
        se::DeviceMemoryAllocator* device_allocator) {}

    StatusOr<std::unique_ptr<Executable>> VsiCompiler::RunBackend(
        std::unique_ptr<HloModule> hlo_module, se::StreamExecutor* stream_exec,
        se::DeviceMemoryAllocator* device_allocator) {}

    StatusOr<std::vector<std::unique_ptr<Executable>>> VsiCompiler::Compile(
        std::unique_ptr<HloModuleGroup> module_group,
        std::vector<std::vector<se::StreamExecutor*>> stream_exec,
        se::DeviceMemoryAllocator* device_allocator) {}

    StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
    VsiCompiler::CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                        const AotCompilationOptions& aot_options) {}

    HloCostAnalysis::ShapeSizeFunction VsiCompiler::ShapeSizeBytesFunction() const {}
    se::Platform::Id VsiCompiler::PlatformId() const {}

} // namespace vsiplugin
} // namespace xla

static bool InitModule() {
    xla::Compiler::RegisterCompilerFactory(
        xla::vsiplugin::kVsiPlatformId,
        []() { return absl::make_unique<xla::vsiplugin::VsiCompiler>(); });

    xla::ComputationPlacer::RegisterComputationPlacer(
        xla::vsiplugin::kVsiPlatformId,
        []() { return absl::make_unique<xla::ComputationPlacer>(); });
    return true;
}
static bool module_initialized = InitModule();
