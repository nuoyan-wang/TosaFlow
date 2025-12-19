add_mlir_library(TosaFlowTransforms
  Passes.cpp
  TosaConvPoolFusion.cpp
  CreateChipletFromTosa.cpp

  DEPENDS
    TosaFlowPassIncGen
    TosaFlowChipletOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRFuncDialect
    MLIRPass
    MLIRTransforms
    MLIRRewrite
    TosaFlowChipletDialect
)
