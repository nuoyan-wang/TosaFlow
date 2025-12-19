//===----------------------------------------------------------------------===//
//
// Chiplet dialect for TosaFlow.
//
//===----------------------------------------------------------------------===//

#ifndef TOSA_FLOW_DIALECT_CHIPLET_TD
#define TOSA_FLOW_DIALECT_CHIPLET_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def ChipletDialect : Dialect {
  let name = "chiplet";
  let summary = "Chiplet grouping dialect for TosaFlow";
  let description = [{
    This dialect introduces chiplet-level grouping operations that can wrap
    parts of a graph to represent placement onto individual chiplets.
  }];
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let cppNamespace = "::mlir::tosa_flow::chiplet";
}

class ChipletOp<string mnemonic, list<Trait> traits = []> :
    Op<ChipletDialect, mnemonic, traits>;

include "tosa-flow/Dialect/Chiplet/Ops.td"

#endif // TOSA_FLOW_DIALECT_CHIPLET_TD
