add_mlir_library(TosaFlowTransforms
  Passes.cpp
  TosaConvPoolFusion.cpp

  DEPENDS
    TosaFlowPassIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRFuncDialect
    MLIRPass
    MLIRTransforms
    MLIRRewrite
  )