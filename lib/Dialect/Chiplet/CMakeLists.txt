add_mlir_library(TosaFlowChipletDialect
  
Chiplet.cpp
Utils.cpp
  
DEPENDS
    TosaFlowChipletOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
)
