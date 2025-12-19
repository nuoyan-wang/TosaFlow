//===----------------------------------------------------------------------===//
//
// TosaFlow InitAllDialects
//
//===----------------------------------------------------------------------===//

#ifndef TOSAFLOW_INITALLDIALECTS_H
#define TOSAFLOW_INITALLDIALECTS_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "tosa-flow/Dialect/Chiplet/Chiplet.h"

namespace mlir {
namespace tosa_flow {

/// Add all dialects needed by TosaFlow to the given registry.
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  // clang-format off
  registry.insert<
    mlir::func::FuncDialect,
    mlir::tosa::TosaDialect,
    mlir::tensor::TensorDialect,
    mlir::linalg::LinalgDialect,
    mlir::memref::MemRefDialect,
    mlir::bufferization::BufferizationDialect,
    mlir::AffineDialect,
    mlir::math::MathDialect,
    mlir::arith::ArithDialect,
    mlir::vector::VectorDialect,
    mlir::scf::SCFDialect,
    mlir::tosa_flow::chiplet::ChipletDialect,
    mlir::LLVM::LLVMDialect,
    mlir::DLTIDialect,
    mlir::ml_program::MLProgramDialect
  >();
  // clang-format on
}

} // namespace tosa_flow
} // namespace mlir

#endif // TOSAFLOW_INITALLDIALECTS_H
