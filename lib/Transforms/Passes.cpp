#include "tosa-flow/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace tosa_flow {

void registerTosaFlowPasses() {
  PassPipelineRegistration<> pipeline(
      "tosa-flow-pipeline",
      "Run canonicalization + conv/pool fusion",
      [](OpPassManager &pm) {
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createTosaConvPoolFusionPass());
        // pm.addPass(createCSEPass());
        // pm.addPass(createCanonicalizerPass());
      });
}

} // namespace tosa_flow
