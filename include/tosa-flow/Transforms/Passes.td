include "mlir/Pass/PassBase.td"

def TosaConvPoolFusionPass : Pass<"tosa-fuse-conv-pool"> {
  let summary = "Fuse TOSA conv2d + clamp + pool patterns";
  let constructor = "tosa_flow::createTosaConvPoolFusionPass()";
}
