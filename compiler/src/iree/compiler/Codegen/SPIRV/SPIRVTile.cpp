// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- SPIRVTile.cpp ------------------------------------------------------===//
//
// This pass tiles and Linalg ops with tensor semantics to invocations.
//
//===----------------------------------------------------------------------===//

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using mlir::iree_compiler::IREE::LinalgExt::TilingPatterns;

#define DEBUG_TYPE "iree-spirv-tile"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Tiling and fusion utilities
//===----------------------------------------------------------------------===//

/// Collects computation ops which we will use as anchor to tile and fuse.
static LogicalResult collectComputeOps(
    func::FuncOp funcOp, SmallVectorImpl<Operation *> &computeOps) {
  // If there are `scf.if` ops, we have both a fast and slow paths for
  // padding handling. Then we need to scan both regions to discover such
  // computation ops so that we can tile and fuse both regions.
  SmallVector<scf::IfOp, 1> ifOps;
  funcOp.walk([&ifOps](scf::IfOp ifOp) { ifOps.push_back(ifOp); });

  if (ifOps.empty()) {
    if (failed(getComputeOps(funcOp, computeOps))) {
      return funcOp.emitOpError("does not contain compute ops");
    }
    while (computeOps.size() > 1) computeOps.erase(computeOps.begin());
    return success();
  }

  if (ifOps.size() > 1) {
    return funcOp.emitError("expected to contain no more than one scf.if ops");
  }

  for (Operation &op : llvm::reverse(*ifOps.front().thenBlock())) {
    if (isa<linalg::LinalgOp, TilingInterface>(op)) {
      computeOps.push_back(&op);
      break;
    }
  }
  if (Block *elseBlock = ifOps.front().elseBlock()) {
    for (Operation &op : llvm::reverse(*elseBlock)) {
      if (isa<linalg::LinalgOp, TilingInterface>(op)) {
        computeOps.push_back(&op);
        break;
      }
    }
  }
  return success();
}

static LogicalResult tileAndDistributeToThreads(linalg::LinalgOp consumerOp) {
  MLIRContext *context = consumerOp.getContext();
  OpBuilder builder(context);
  SmallVector<int64_t> tileSizes = getTileSizes(consumerOp, 1);
  auto identityLoopOrder =
      llvm::to_vector<4>(llvm::seq<int64_t>(0, tileSizes.size()));

  FailureOr<linalg::TileLoopNest> loopNest =
      IREE::LinalgExt::tileConsumerAndFuseProducers(
          builder, consumerOp, tileSizes, identityLoopOrder, llvm::None);
  if (failed(loopNest)) {
    return consumerOp.emitOpError("failed tiling and fusing producers");
  }

  consumerOp->replaceAllUsesWith(loopNest->getRootOpReplacementResults());

  // We don't distribute here; instead, it will be done in a later step
  // after bufferization. So add attributes to the tiled loop nest to
  // indicate that they should be distributed to invocations.
  ArrayRef<scf::ForOp> loops = loopNest->getLoopOps();
  assert(loops.size() <= kNumGPUDims);
  const char *attrName = getSPIRVDistributeAttrName();
  for (int i = loops.size() - 1, dim = 0; i >= 0; --i) {
    loops[i]->setAttr(attrName, builder.getIndexAttr(dim++));
  }
  return success();
}

/// Populates `patterns` with patterns that tiles convolution/matmul ops with
/// markers.
static void populateTilingReductionPatterns(RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  auto getTileSizeFn = [&](OpBuilder &builder, Operation *op) {
    return getTileSizes(builder, op, 2);
  };
  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizeComputationFunction(getTileSizeFn);
  auto marker = StringAttr::get(context, getTileReductionMarker());
  auto filter =
      IREE::LinalgExt::LinalgTransformationFilter({marker}, llvm::None);

  TilingPatterns<linalg::BatchMatmulOp, linalg::Conv2DNchwFchwOp,
                 linalg::Conv2DNhwcHwcfOp, linalg::DepthwiseConv2DNhwcHwcOp,
                 linalg::GenericOp, linalg::MatmulOp>::insert(patterns,
                                                              tilingOptions,
                                                              filter);
}

/// Tiles reduction dimensions.
static LogicalResult tileReduction(func::FuncOp funcOp) {
  MLIRContext *context = funcOp.getContext();

  // Set markers to drive tiling reduction dimensions.
  OpBuilder builder(context);
  auto marker = builder.getStringAttr(getTileReductionMarker());
  funcOp.walk([&](linalg::LinalgOp op) {
    if (isa<linalg::ContractionOpInterface>(*op) ||
        isa<linalg::ConvolutionOpInterface>(*op) ||
        isa<linalg::GenericOp>(*op)) {
      op->setAttr(IREE::LinalgExt::LinalgTransforms::kLinalgTransformMarker,
                  marker);
    }
  });

  RewritePatternSet patterns(context);
  populateTilingReductionPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    return funcOp.emitError("failed tiling reduction dimensions");
  }

  LLVM_DEBUG({
    llvm::dbgs() << "--- After tiling reduction dimensions  ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });
  return success();
}

/// Fuses `tensor.pad` ops into the the materalized loop nests containing
/// their consumer ops.
static void fusePadIntoConsumer(func::FuncOp funcOp) {
  MLIRContext *context = funcOp.getContext();
  RewritePatternSet patterns(context);
  patterns.insert<linalg::ExtractSliceOfPadTensorSwapPattern>(
      context, [](tensor::ExtractSliceOp) { return false; });
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));

  LLVM_DEBUG({
    llvm::dbgs() << "--- After fusing padding into consumers ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });
};

/// Concretizes `tensor.pad` ops' result shapes.
static void concretizePadShape(func::FuncOp funcOp) {
  MLIRContext *context = funcOp.getContext();
  RewritePatternSet patterns(context);
  populateConcretizePadResultShapePatterns(context, patterns);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));

  LLVM_DEBUG({
    llvm::dbgs() << "--- After concretizing pad result shape ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });
}

/// Tiles one of the convolution output window dimensions with size 1 to prepare
/// for downsizing 2-D convolution ops into 1-D ones.
static LogicalResult tileAndUnrollConvWindow(func::FuncOp funcOp) {
  SmallVector<linalg::ConvolutionOpInterface, 1> convOps;
  funcOp.walk([&convOps](linalg::ConvolutionOpInterface convOp) {
    convOps.push_back(convOp);
  });

  for (linalg::ConvolutionOpInterface convOp : convOps) {
    auto consumerOp = cast<linalg::LinalgOp>(*convOp);
    OpBuilder builder(funcOp.getContext());
    SmallVector<int64_t> tileSizes = getTileSizes(consumerOp, 3);
    auto identityLoopOrder =
        llvm::to_vector<4>(llvm::seq<int64_t>(0, tileSizes.size()));

    FailureOr<linalg::TileLoopNest> loopNest =
        IREE::LinalgExt::tileConsumerAndFuseProducers(
            builder, consumerOp, tileSizes, identityLoopOrder, llvm::None);
    if (failed(loopNest)) {
      return consumerOp.emitOpError("failed tiling and fusing producers");
    }

    consumerOp->replaceAllUsesWith(loopNest->getRootOpReplacementResults());

    // Fully unroll the generated loop. This allows us to remove the loop
    // for parallel output window dimension, so it helps future vector
    // transformations.
    if (!loopNest->getLoopOps().empty()) {
      assert(loopNest->getLoopOps().size() == 1);
      scf::ForOp loopOp = loopNest->getLoopOps().front();
      IntegerAttr ub;
      if (!matchPattern(loopOp.getUpperBound(), m_Constant(&ub))) {
        return loopOp.emitOpError("upper bound should be a constant");
      }
      if (failed(mlir::loopUnrollByFactor(loopOp, ub.getInt()))) {
        return loopOp.emitOpError("failed unrolling by factor 1");
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After tiling convolution output window ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Main pass
//===----------------------------------------------------------------------===//

namespace {

class SPIRVTilePass final : public SPIRVTileBase<SPIRVTilePass> {
 public:
  SPIRVTilePass() = default;
  SPIRVTilePass(const SPIRVTilePass &pass) = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    func::FuncOp funcOp = getOperation();

    // Try to find computation ops which we will use as anchor to tile and fuse.
    SmallVector<Operation *> computeOps;
    if (failed(collectComputeOps(funcOp, computeOps)))
      return signalPassFailure();
    assert(computeOps.size() <= 2);

    // Now tile the last computation op to invocations and fuse all operand
    // computation ops into the materialized loop nest.
    for (Operation *computeOp : computeOps) {
      auto consumerOp = dyn_cast<linalg::LinalgOp>(computeOp);
      if (failed(tileAndDistributeToThreads(consumerOp)))
        return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After tiling to invocations ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    fusePadIntoConsumer(funcOp);

    concretizePadShape(funcOp);

    if (failed(tileReduction(funcOp))) return signalPassFailure();

    fusePadIntoConsumer(funcOp);

    if (failed(tileAndUnrollConvWindow(funcOp))) return signalPassFailure();

    concretizePadShape(funcOp);

    {  // Downsize n-D (n > 1) convolutions to 1-D.
      RewritePatternSet patterns(context);
      linalg::populateDecomposeConvolutionPatterns(patterns);
      // Downsizing creates consecutive extract/insert slice ops. Merge them.
      tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
      // Pull in patterns to fold constant insert/extract slice op parameters.
      tensor::InsertSliceOp::getCanonicalizationPatterns(patterns, context);
      tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, context);
      // Pull in scf.for op canonicalization patterns to help hoisting across
      // multiple loops and remove loop carried values unused in the body.
      scf::ForOp::getCanonicalizationPatterns(patterns, context);
      (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));

      LLVM_DEBUG({
        llvm::dbgs() << "--- After Downsizing N-D convolution to 1-D  ---\n";
        funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
        llvm::dbgs() << "\n\n";
      });
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createSPIRVTilePass() {
  return std::make_unique<SPIRVTilePass>();
}

}  // namespace iree_compiler
}  // namespace mlir