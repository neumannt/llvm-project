//===-- Passes.td - StandardOps pass definition file -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_STANDARD_TRANSFORMS_PASSES
#define MLIR_DIALECT_STANDARD_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def FuncBufferize : Pass<"func-bufferize", "ModuleOp"> {
  let summary = "Bufferize func/call/return ops";
  let description = [{
    A bufferize pass that bufferizes std.func and std.call ops.

    Because this pass updates std.func ops, it must be a module pass. It is
    useful to keep this pass separate from other bufferizations so that the
    other ones can be run at function-level in parallel.

    This pass must be done atomically because it changes func op signatures,
    which requires atomically updating calls as well throughout the entire
    module.

    This pass also changes the type of block arguments, which requires that all
    successor arguments of predecessors be converted. This is achieved by
    rewriting terminators based on the information provided by the
    `BranchOpInterface`.
    As this pass rewrites function operations, it also rewrites the
    corresponding return operations. Other return-like operations that
    implement the `ReturnLike` trait are not rewritten in general, as they
    require that the corresponding parent operation is also rewritten.
    Finally, this pass fails for unknown terminators, as we cannot decide
    whether they need rewriting.
  }];
  let constructor = "mlir::createFuncBufferizePass()";
  let dependentDialects = ["bufferization::BufferizationDialect",
                           "memref::MemRefDialect"];
}

#endif // MLIR_DIALECT_STANDARD_TRANSFORMS_PASSES
