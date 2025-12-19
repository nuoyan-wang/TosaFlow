//===----------------------------------------------------------------------===//
// Chiplet dialect entry
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "tosa-flow/Dialect/Chiplet/ChipletDialect.h.inc"

#define GET_OP_CLASSES
#include "tosa-flow/Dialect/Chiplet/Chiplet.h.inc"
