//===----------------------------------------------------------------------===//
//
// Chiplet dialect operations for TosaFlow.
//
//===----------------------------------------------------------------------===//

#ifndef TOSA_FLOW_DIALECT_CHIPLET_OPS_TD
#define TOSA_FLOW_DIALECT_CHIPLET_OPS_TD

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "tosa-flow/Dialect/Chiplet/Chiplet.td"

//===----------------------------------------------------------------------===//
// Chiplet Ops
//===----------------------------------------------------------------------===//

// Group tasks assigned to one chiplet.
def ChipOp : ChipletOp<"chip",
    [SingleBlockImplicitTerminator<"YieldOp">,
     ParentOneOf<["::mlir::func::FuncOp"]>]> {
  let summary = "Chiplet chip grouping tasks";
  let description = [{
    Groups one or more chiplet.task ops intended to execute on the same chiplet.
    The body is a single block terminated by chiplet.yield.
  }];

  let arguments = (ins I32Attr:$id); // numeric chiplet ID
  let results = (outs Variadic<AnyType>:$results);
  let regions   = (region SizedRegion<1>:$body);
  let assemblyFormat = "$id attr-dict-with-keyword ( `:` type($results)^ )? $body";
}

// Task within a chiplet chip.
def TaskOp : ChipletOp<"task",
    [SingleBlockImplicitTerminator<"YieldOp">,
     HasParent<"ChipOp">]> {
  let summary = "Task inside a chiplet chip";
  let description = [{
    Contains a sequence of operations to run on a chiplet; single-block region
    terminated by chiplet.yield.
  }];

  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword ( `:` type($results)^ )? $body";
}

// Terminator for chip/task regions.
def YieldOp : ChipletOp<"yield",
    [NoMemoryEffect, ReturnLike, Terminator,
     ParentOneOf<["ChipOp", "TaskOp"]>]> {
  let summary = "Terminate and yield results of a chip or task";

  let arguments = (ins Variadic<AnyType>:$results);
  let assemblyFormat = "$results attr-dict `:` type($results)";
  
  let builders = [OpBuilder<(ins), "build($_builder, $_state, llvm::None);">];
}

#endif // TOSA_FLOW_DIALECT_CHIPLET_OPS_TD
