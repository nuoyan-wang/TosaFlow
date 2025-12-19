#include "tosa-flow/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::tosa_flow {
#define GEN_PASS_DEF_TOSACONVPOOLFUSIONPASS
#define GEN_PASS_DEF_CREATECHIPLETFROMTOSA
#include "tosa-flow/Transforms/Passes.h.inc"

struct TosaFlowPipelineOptions
    : public PassPipelineOptions<TosaFlowPipelineOptions> {
  Option<unsigned> debugPoint{*this, "debug-point", ::llvm::cl::init(0),
                              "Stop the pipeline at the given debug point"};
  Option<unsigned> numChiplets{*this, "num-chiplets", ::llvm::cl::init(2),
                               "Number of chiplets to spread ops across"};
  Option<std::string> chipletLoads{
      *this, "chiplet-loads", ::llvm::cl::init(""),
      "Comma-separated chipletId=count pairs (overrides num-chiplets)"};
};

void registerTosaFlowPasses() {
  PassPipelineRegistration<TosaFlowPipelineOptions>(
      "tosa-flow-pipeline", "Run canonicalization + conv/pool fusion",
      [](OpPassManager &pm, const TosaFlowPipelineOptions &opts) {
        // pm.addPass(createCanonicalizerPass());
        // if (opts.debugPoint == 1) return;

        // pm.nest<func::FuncOp>().addPass(createTosaConvPoolFusionPass());
        // if (opts.debugPoint == 2) return;

        pm.nest<func::FuncOp>().addPass(
            createCreateChipletFromTosaPass(opts.numChiplets, opts.chipletLoads));
        if (opts.debugPoint == 3) return;

        // pm.addPass(createCSEPass());
        // if (opts.debugPoint == 4) return;

        // pm.addPass(createCanonicalizerPass());
      });
}
} // namespace mlir::tosa_flow
