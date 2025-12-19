include "mlir/Pass/PassBase.td"

def TosaConvPoolFusionPass : Pass<"tosa-fuse-conv-pool"> {
  let summary = "Fuse TOSA conv2d + clamp + pool patterns";
  let constructor = "mlir::tosa_flow::createTosaConvPoolFusionPass()";
}

def CreateChipletFromTosa :
      Pass<"tosa-create-chiplet-from-tosa", "func::FuncOp"> {
  let summary = "Outline selected TOSA ops into chiplet.task/chip";
  let constructor = "mlir::tosa_flow::createCreateChipletFromTosaPass()";
  let options = [
    Option<"numChiplets","num-chiplets","unsigned","1",
           "Number of chiplets to spread ops across (ignored if chiplet-loads is set)">,
    Option<"chipletLoads","chiplet-loads","std::string","",
           "Comma-separated chipletId=count pairs, e.g. 0=5,1=3,2=4">
  ];
}
