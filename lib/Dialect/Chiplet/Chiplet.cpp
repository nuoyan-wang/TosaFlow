#include "tosa-flow/Dialect/Chiplet/Chiplet.h"
#include "tosa-flow/Dialect/Chiplet/ChipletDialect.cpp.inc"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/Support/ErrorHandling.h"

#define GET_OP_CLASSES
#include "tosa-flow/Dialect/Chiplet/ChipletOps.cpp.inc"

using namespace mlir;
using namespace mlir::tosa_flow::chiplet;

Attribute ChipletDialect::parseAttribute(DialectAsmParser &parser, Type type) const {
  parser.emitError(parser.getNameLoc(), "unknown chiplet attribute");
  return {};
}

void ChipletDialect::printAttribute(Attribute attr, DialectAsmPrinter &os) const {
  llvm_unreachable("no chiplet attributes");
}

Type ChipletDialect::parseType(DialectAsmParser &parser) const {
  parser.emitError(parser.getNameLoc(), "unknown chiplet type");
  return {};
}

void ChipletDialect::printType(Type type, DialectAsmPrinter &os) const {
  llvm_unreachable("no chiplet types");
}

void ChipletDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tosa-flow/Dialect/Chiplet/ChipletOps.cpp.inc"
  >();
}
