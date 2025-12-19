#pragma once
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <string>

namespace mlir {
namespace tosa_flow {
#define GEN_PASS_DECL_TOSACONVPOOLFUSIONPASS
#define GEN_PASS_DECL_CREATECHIPLETFROMTOSA
#include "tosa-flow/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createTosaConvPoolFusionPass();
std::unique_ptr<mlir::Pass>
createCreateChipletFromTosaPass(unsigned numChiplets = 1, std::string chipletLoads = "");
void registerTosaFlowPasses();
} // namespace tosa_flow
} // namespace mlir
