add_mlir_dialect_library(MLIRStandard
  IR/Ops.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/StandardOps

  DEPENDS
  MLIRStandardOpsIncGen

  LINK_LIBS PUBLIC
  MLIRArithmetic
  MLIRCallInterfaces
  MLIRCastInterfaces
  MLIRControlFlow
  MLIRControlFlowInterfaces
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRVectorInterfaces
  )

add_subdirectory(Transforms)
