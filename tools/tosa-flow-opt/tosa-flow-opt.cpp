//===----------------------------------------------------------------------===//
//
// TosaFlow Optimization Tool
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "tosa-flow/InitAllDialects.h"
#include "tosa-flow/InitAllPasses.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  mlir::tosa_flow::registerAllDialects(registry);
  mlir::tosa_flow::registerAllPasses();

  return mlir::failed(mlir::MlirOptMain(
      argc, argv, "TosaFlow Optimization Tool", registry, /*allowUnregisteredDialects=*/true));
}
