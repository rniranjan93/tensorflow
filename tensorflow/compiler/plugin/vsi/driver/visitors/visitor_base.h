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

#ifndef TENSORFLOW_COMPILER_PLUGIN_VSI_DRIVER_VISITORS_VISITOR_BASE_H_
#define TENSORFLOW_COMPILER_PLUGIN_VSI_DRIVER_VISITORS_VISITOR_BASE_H_

#include <string>
#include <unordered_map>
#include "tensorflow/compiler/plugin/vsi/driver/vsi_executor.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/tensor.h"
#include "tim/vx/types.h"
namespace xla {
namespace vsiplugin {

/*
 * The base visitor handles all operations that are element-wise.
 * This includes all explicitly element-wise ops, for temprarily, they
 * are implemented by hlo_evaluator, and repalce it with AIM implment
 * step by step. All of these have no element to element dependencies.
 */
class BaseVisitor : public DfsHloVisitor {
 public:
  BaseVisitor(VsiExecutor* executor) : executor_(executor),
   graph_(executor->getGraph()) {};

  std::shared_ptr<tim::vx::Tensor> createTensorFromShape(const Shape &shape,
    tim::vx::TensorAttribute attr = tim::vx::TensorAttribute::INPUT){
    tim::vx::ShapeType timShape;
    tim::vx::Quantization timQuant;
    if(shape.is_static()){
        for( auto d : shape.dimensions())
        timShape.push_back(d);
    }
    auto type = convertTfPrimitiveTypeToTim(shape.element_type());
    if(type != tim::vx::DataType::FLOAT32 &&
       type != tim::vx::DataType::FLOAT16) {
         LOG(FATAL)<< "NOT implement";
       }
    tim::vx::TensorSpec timSpec(type, timShape,
                attr, timQuant);
    return graph_->CreateTensor(timSpec);
  }

  static tim::vx::DataType convertTfPrimitiveTypeToTim(xla::PrimitiveType xlaType){
      switch(xlaType){
        case S8:{
          return tim::vx::DataType::INT8;
        }
        case S16:{
          return tim::vx::DataType::INT16;
        }
        case S32:{
          return tim::vx::DataType::INT32;
        }
        case F32:{
          return tim::vx::DataType::FLOAT32;
        }
        default:
          LOG(FATAL)<<"not supported datat type";
      }
  }
  virtual const Shape& GetOutputShape(HloInstruction*) const;

    Literal evaluate(const HloComputation& computation
         /*absl::Span<const Literal* const> arg_literals*/);
    
    std::shared_ptr<tim::vx::Tensor> evaluate(const HloComputation& computation,
        std::vector<Literal>& argument_literals);

    Status HandleHloOp(HloInstruction* hlo);

    Status FinishVisit(HloInstruction* root) final;

    // Returns the already-evaluated literal result for the instruction.
    //
    // A Constant instruction is considered evaluated and its literal will be
    // returned directly without looking up the cache.
    //
    // Similarly, a Parameter instruction is considered evaluated and its literal
    // is looked up in arg_literals.
    //
    // Crash with log if the given instruction has not been evaluated previously.
    const Literal& GetEvaluatedLiteralFor(const HloInstruction* hlo) {
        if (hlo->IsConstant()) {
            return hlo->literal();
        }
        // if (hlo->opcode() == HloOpcode::kParameter) {
        //     return *arg_literals_.at(hlo->parameter_number());
        // }
        auto it = evaluated_.find(hlo);
        CHECK(it != evaluated_.end())
            << "could not find evaluated value for: " << hlo->ToString();
        return it->second;
    }
    const std::shared_ptr<tim::vx::Tensor> GetEvaluatedTensorFor(const HloInstruction* hlo) {
        auto it = evaluatedDevMem_.find(hlo);
        CHECK(it != evaluatedDevMem_.end())
            << "could not find evaluated value for: " << hlo->ToString();
        return executor_->getTensor( it->second );
    }

  // Called by HandleElementwiseBinarythe FinishVisit.
  virtual Status FinishScopedVisit(HloInstruction* root) {
    return Status::OK();
  }
  Status HandleElementwiseBinary(HloInstruction* hlo) override;

  Status HandleConstant(HloInstruction* hlo) override;

  Status HandleParameter(HloInstruction* hlo) override;

  Status HandleReshape(HloInstruction* hlo) override;

#define HANDLE_AS_HLO_OP(Name) \
  Status Name(HloInstruction* inst) override { return HandleHloOp(inst); }

  /*
   * Operations not processed by this visitor.
   */
#define UNIMPLEMENTED(Name) \
  Status Name(HloInstruction* inst) override { \
    LOG(INFO)<< "@@ unimplement instruction "<<__FUNCTION__; \
    return Unimplemented(inst); \
    };

  UNIMPLEMENTED(HandleTupleSelect)
  UNIMPLEMENTED(HandleConvert)
  UNIMPLEMENTED(HandleCollectivePermuteStart)
  UNIMPLEMENTED(HandleCollectivePermuteDone)
  UNIMPLEMENTED(HandleRngBitGenerator)
  UNIMPLEMENTED(HandleBitcastConvert)
  UNIMPLEMENTED(HandleAllReduce)
  UNIMPLEMENTED(HandleAllGather)
  UNIMPLEMENTED(HandleGetTupleElement)
  UNIMPLEMENTED(HandleFusion)
  UNIMPLEMENTED(HandleCall)
  UNIMPLEMENTED(HandleCustomCall)
  UNIMPLEMENTED(HandleTuple)
  UNIMPLEMENTED(HandleMap)
  UNIMPLEMENTED(HandleConditional)
  UNIMPLEMENTED(HandleInfeed)
  UNIMPLEMENTED(HandleAfterAll)
  UNIMPLEMENTED(HandleReal)
  UNIMPLEMENTED(HandleAllToAll)
  UNIMPLEMENTED(HandleAddDependency)
  UNIMPLEMENTED(HandleElementwiseUnary)
  UNIMPLEMENTED(HandleClamp)
  UNIMPLEMENTED(HandleSelect)
  UNIMPLEMENTED(HandleCompare)
  UNIMPLEMENTED(HandleRng)
  UNIMPLEMENTED(HandleSlice)
  UNIMPLEMENTED(HandleDynamicSlice)
  UNIMPLEMENTED(HandleDynamicUpdateSlice)
  UNIMPLEMENTED(HandleSelectAndScatter)
  UNIMPLEMENTED(HandleWhile)
  UNIMPLEMENTED(HandlePad)
  UNIMPLEMENTED(HandleReverse)
  UNIMPLEMENTED(HandleSort)
  UNIMPLEMENTED(HandleReduce)
  UNIMPLEMENTED(HandleBitcast)
  UNIMPLEMENTED(HandleBroadcast)
  //UNIMPLEMENTED(HandleReshape)
  UNIMPLEMENTED(HandleTranspose)
  UNIMPLEMENTED(HandleReducePrecision)
  UNIMPLEMENTED(HandleOutfeed)
  UNIMPLEMENTED(HandleSend)
  UNIMPLEMENTED(HandleSendDone)
  UNIMPLEMENTED(HandleRecv)
  UNIMPLEMENTED(HandleRecvDone)
  UNIMPLEMENTED(HandleBatchNormInference)
  UNIMPLEMENTED(HandleBatchNormTraining)
  UNIMPLEMENTED(HandleBatchNormGrad)
  UNIMPLEMENTED(HandleFft)
  UNIMPLEMENTED(HandleGather)
  UNIMPLEMENTED(HandleCopy)
  UNIMPLEMENTED(HandleIota)
  UNIMPLEMENTED(HandleScatter)
  UNIMPLEMENTED(HandleCollectivePermute)
  UNIMPLEMENTED(HandleConcatenate)
  UNIMPLEMENTED(HandleGetDimensionSize)
  UNIMPLEMENTED(HandleReplicaId)
  UNIMPLEMENTED(HandleTriangularSolve)
  UNIMPLEMENTED(HandleCholesky)
  UNIMPLEMENTED(HandlePartitionId)
  UNIMPLEMENTED(HandleRngGetAndUpdateState)
  UNIMPLEMENTED(HandleCopyStart)
  UNIMPLEMENTED(HandleCopyDone)
  UNIMPLEMENTED(HandleSetDimensionSize)
  UNIMPLEMENTED(HandleDot)
  UNIMPLEMENTED(HandleConvolution)
  UNIMPLEMENTED(HandleReduceWindow)

 protected:
  Status Unimplemented(HloInstruction* inst);

  const std::string name_ = "vsi base visitor";

  std::unique_ptr<HloEvaluator> cpu_evaluator_;

private:
    VsiExecutor *executor_;

    // Tracks the HLO instruction and its evaluated literal result.
    // Parameters and constants aren't stored here,
    // TODO: it is better the Literal value was repalced with device memory
    //       handle.
    std::unordered_map<const HloInstruction *, Literal> evaluated_;
    std::unordered_map<const HloInstruction *, int> evaluatedDevMem_;
    std::vector<Literal> arg_literals_;
    std::shared_ptr<tim::vx::Graph> graph_;
};

}  // namespace vsiplugin
}  // namespace xla

#endif
