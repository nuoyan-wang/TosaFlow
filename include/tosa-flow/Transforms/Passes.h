#pragma once

#include "mlir/Pass/Pass.h"

namespace tosa_flow {

std::unique_ptr<mlir::Pass> createTosaConvPoolFusionPass();
void registerTosaFlowPasses();

} // namespace tosa_flow
