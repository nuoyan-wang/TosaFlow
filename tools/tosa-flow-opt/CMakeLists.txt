get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_tool(tosa-flow-opt
  tosa-flow-opt.cpp
)

llvm_update_compile_flags(tosa-flow-opt)

target_link_libraries(tosa-flow-opt
  PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    MLIROptLib

    TosaFlowTransforms

    MLIRFuncDialect
)
