//===- X86CarryFlagReturn.cpp - Prepare returns in the carry flag ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A function with the "x86-carry-flag-return" attribute returns its trailing
// i1 return value in the carry flag (see X86::CarryFlagReturnAttr). SimplifyCFG
// merges all returns into one block, so the flag usually reaches the return
// through a phi or as a branch condition from an earlier block. Instruction
// selection works on one block at a time and would then materialize the flag
// in a register only to move it back into the carry flag at the return.
//
// This pass duplicates such a return block into its predecessors, if it is
// small, and replaces the flag by a constant where it is known there (a phi
// input, or a condition implied by the branch leading there). A constant flag
// is set right before the return with STC or CLC.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86ISelLowering.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;

#define DEBUG_TYPE "x86-carry-flag-return"

namespace {
class X86CarryFlagReturnLegacy : public FunctionPass {
public:
  static char ID;
  X86CarryFlagReturnLegacy() : FunctionPass(ID) {}
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addPreserved<DominatorTreeWrapperPass>();
  }
  StringRef getPassName() const override {
    return "X86 Carry Flag Return Preparation";
  }
};
} // end anonymous namespace

char X86CarryFlagReturnLegacy::ID = 0;

INITIALIZE_PASS(X86CarryFlagReturnLegacy, DEBUG_TYPE,
                "X86 Carry Flag Return Preparation", false, false)

FunctionPass *llvm::createX86CarryFlagReturnLegacyPass() {
  return new X86CarryFlagReturnLegacy();
}

/// The use of the flag that \p RI returns, if it is set by an insertvalue.
static Use *getFlagUse(ReturnInst *RI) {
  auto *STy = dyn_cast_or_null<StructType>(
      RI->getReturnValue() ? RI->getReturnValue()->getType() : nullptr);
  if (!STy || STy->getNumElements() == 0 ||
      !STy->getElementType(STy->getNumElements() - 1)->isIntegerTy(1))
    return nullptr;
  unsigned FlagIdx = STy->getNumElements() - 1;
  Value *V = RI->getReturnValue();
  while (auto *IV = dyn_cast<InsertValueInst>(V)) {
    if (IV->getNumIndices() == 1 && IV->getIndices()[0] == FlagIdx)
      return &IV->getOperandUse(
          InsertValueInst::getInsertedValueOperandIndex());
    V = IV->getAggregateOperand();
  }
  return nullptr;
}

/// Whether the return block \p BB is worth duplicating into its predecessors.
static bool shouldDuplicate(BasicBlock *BB, ReturnInst *RI) {
  if (pred_empty(BB) || BB->hasNPredecessorsOrMore(9) || BB->hasAddressTaken())
    return false;
  // The flag must be set in the block (possibly to a phi), or the value must
  // be a phi.
  Use *Flag = getFlagUse(RI);
  auto *PN = dyn_cast_or_null<PHINode>(RI->getReturnValue());
  if (Flag ? isa<Constant>(Flag->get()) : !(PN && PN->getParent() == BB))
    return false;
  // Only phis and the construction of the return value.
  unsigned NumInsts = 0;
  for (Instruction &I : *BB) {
    if (isa<PHINode>(I) || I.isDebugOrPseudoInst())
      continue;
    if (!isa<InsertValueInst, ExtractValueInst, ReturnInst>(I) ||
        ++NumInsts > 8)
      return false;
  }
  for (BasicBlock *Pred : predecessors(BB))
    if (!isa<UncondBrInst, CondBrInst>(Pred->getTerminator()))
      return false;
  return true;
}

/// The value of \p Cond at \p I if it is implied by a dominating branch.
static std::optional<bool> getImpliedValue(Value *Cond, Instruction *I,
                                           const DataLayout &DL) {
  // CodeGenPrepare freezes the conditions of branches that it creates from
  // selects.
  BasicBlock *BB = I->getParent();
  BasicBlock *Pred = BB->getSinglePredecessor();
  auto *Br = Pred ? dyn_cast<CondBrInst>(Pred->getTerminator()) : nullptr;
  if (Br && Br->getSuccessor(0) != Br->getSuccessor(1)) {
    Value *BrCond = Br->getCondition();
    if (auto *Fr = dyn_cast<FreezeInst>(BrCond))
      BrCond = Fr->getOperand(0);
    if (BrCond == Cond)
      return Br->getSuccessor(0) == BB;
  }
  return isImpliedByDomCondition(Cond, I, DL);
}

/// Copy the return block \p BB into \p Pred, which ends in an unconditional
/// branch to it. Returns the new return instruction.
static ReturnInst *duplicateReturnInto(BasicBlock *BB, BasicBlock *Pred,
                                       DominatorTree *DT) {
  ValueToValueMapTy VMap;
  for (PHINode &PN : BB->phis())
    VMap[&PN] = PN.getIncomingValueForBlock(Pred);
  Instruction *Br = Pred->getTerminator();
  ReturnInst *NewRet = nullptr;
  for (Instruction &I : *BB) {
    if (isa<PHINode>(I))
      continue;
    Instruction *C = I.clone();
    C->insertBefore(Br->getIterator());
    RemapInstruction(C, VMap, RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);
    VMap[&I] = C;
    if (auto *RI = dyn_cast<ReturnInst>(C))
      NewRet = RI;
  }
  BB->removePredecessor(Pred);
  Br->eraseFromParent();
  if (DT)
    DT->deleteEdge(Pred, BB);
  return NewRet;
}

bool X86CarryFlagReturnLegacy::runOnFunction(Function &F) {
  if (skipFunction(F) || !F.hasFnAttribute(X86::CarryFlagReturnAttr))
    return false;
  const DataLayout &DL = F.getDataLayout();
  auto *DTWP = getAnalysisIfAvailable<DominatorTreeWrapperPass>();
  DominatorTree *DT = DTWP ? &DTWP->getDomTree() : nullptr;

  SmallVector<std::pair<BasicBlock *, ReturnInst *>, 4> Returns;
  for (BasicBlock &BB : F)
    if (auto *RI = dyn_cast<ReturnInst>(BB.getTerminator()))
      if (shouldDuplicate(&BB, RI))
        Returns.push_back({&BB, RI});

  bool Changed = false;
  for (auto [BB, RI] : Returns) {
    (void)RI;
    SmallVector<BasicBlock *, 4> Preds(predecessors(BB));
    SmallVector<ReturnInst *, 4> NewRets;
    for (BasicBlock *Pred : Preds) {
      // Give each edge a block of its own that ends in an unconditional
      // branch to BB, into which the return is folded.
      BasicBlock *From = Pred;
      if (!isa<UncondBrInst>(Pred->getTerminator()))
        From = SplitEdge(Pred, BB, DT);
      NewRets.push_back(duplicateReturnInto(BB, From, DT));
    }
    // Where the flag is implied by the branch leading to the return, use the
    // constant. The insertvalue that sets the flag may be shared with other
    // paths, so set the constant in a new one right before the return.
    for (ReturnInst *NewRI : NewRets) {
      Use *Flag = getFlagUse(NewRI);
      if (!Flag || isa<Constant>(Flag->get()))
        continue;
      if (std::optional<bool> Known = getImpliedValue(Flag->get(), NewRI, DL)) {
        Value *RetVal = NewRI->getReturnValue();
        unsigned FlagIdx =
            cast<StructType>(RetVal->getType())->getNumElements() - 1;
        NewRI->setOperand(
            0, InsertValueInst::Create(
                   RetVal, ConstantInt::getBool(F.getContext(), *Known),
                   FlagIdx, "", NewRI->getIterator()));
      }
    }
    if (pred_empty(BB)) {
      // Deleting the last edge usually removed it from the tree already.
      if (DT && DT->getNode(BB))
        DT->eraseNode(BB);
      DeleteDeadBlock(BB);
    }
    Changed = true;
  }
  return Changed;
}
