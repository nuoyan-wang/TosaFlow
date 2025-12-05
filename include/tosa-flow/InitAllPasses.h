//===----------------------------------------------------------------------===//
//
// TosaFlow InitAllPasses
//
//===----------------------------------------------------------------------===//

#ifndef TOSAFLOW_INITALLPASSES_H
#define TOSAFLOW_INITALLPASSES_H

#include "mlir/InitAllPasses.h"
#include "tosa-flow/Transforms/Passes.h"

namespace mlir {
namespace tosa_flow {


inline void registerAllPasses() {
  mlir::registerAllPasses();
  ::tosa_flow::registerTosaFlowPasses();
}

} // namespace tosa_flow
} // namespace mlir

#endif // TOSAFLOW_INITALLPASSES_H
