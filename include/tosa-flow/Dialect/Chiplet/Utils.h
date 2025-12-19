//===----------------------------------------------------------------------===//
// Chiplet dialect utilities.
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "tosa-flow/Dialect/Chiplet/Chiplet.h"

namespace mlir::tosa_flow::chiplet {

TaskOp fuseOpsIntoTask(ArrayRef<Operation *> ops, PatternRewriter &rewriter,
                       int64_t chipletId = 0, bool insertToLastOp = false);

} // namespace mlir::tosa_flow::chiplet
