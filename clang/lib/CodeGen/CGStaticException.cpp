//===--- CGStaticException.cpp - P0709 static exceptions ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements code generation for the prototype of P0709
// "Zero-overhead deterministic exceptions: Throwing values".
//
// A function declared 'throws' receives a hidden pointer to a caller-owned
// std::error object (see ClangToLLVMArgMapping). To fail, it stores the
// error there and returns normally. The first member of std::error, its
// domain pointer, is never null for a valid error, so the caller tests it
// after the call to find out whether the callee failed. The caller clears the
// domain pointer before the call, except when it simply forwards its own
// out-parameter (propagation needs no copying at all).
//
// A static exception is delivered to a "target" (StaticErrorTarget): the
// innermost enclosing try statement with a catch(std::error) or catch(...)
// handler, otherwise the caller of a 'throws' function, otherwise (in a
// function that is not 'throws') it is thrown as a dynamic exception via
// std::__throw_error_as_dynamic. Delivering to a local handler or to the
// caller is a plain forward branch; all cleanups (including EH-only ones like
// destroying partially constructed objects) are emitted inline on that path.
//
// With -fstatic-exceptions-abi=register or carry, a 'throws' function instead
// returns either its value or the error in the return registers, plus a flag
// that tells which (see StaticErrorReturnLayout). Inside the function, errors
// still go through a std::error slot: a local that is returned from the error
// return block. After a call, the caller tests the flag and stores the error
// to the slot of the target.
//
// Dynamic exceptions that escape the body of a 'throws' function are caught
// by an implicit catch(...) and translated with
// std::__error_from_current_exception. Hence 'throws' functions are nounwind
// and neither they nor their callers need unwind tables or landing pads for
// them.
//
//===----------------------------------------------------------------------===//

#include "CGCXXABI.h"
#include "CGCleanup.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/StmtCXX.h"
#include "clang/CodeGen/CGFunctionInfo.h"
#include "llvm/IR/MDBuilder.h"

using namespace clang;
using namespace CodeGen;

bool CodeGenFunction::isStaticThrowsFunction() const {
  return CurFnInfo && CurFnInfo->hasStaticErrorParam();
}

llvm::Type *CodeGenFunction::getStaticErrorLLVMType() {
  return ConvertTypeForMem(getContext().getStdErrorType());
}

CharUnits CodeGenFunction::getStaticErrorAlignment() {
  return getContext().getTypeAlignInChars(getContext().getStdErrorType());
}

Address CodeGenFunction::makeStaticErrorAddress(llvm::Value *Ptr) {
  if (!getContext().getStdErrorDecl()) {
    // Sema guarantees std::error is known whenever 'throws' is used, but be
    // defensive (e.g. for declarations imported from an AST file).
    CGM.ErrorUnsupported(CurFuncDecl, "'throws' without std::error");
    return Address(Ptr, llvm::ArrayType::get(Int8PtrTy, 2), getPointerAlign());
  }
  return Address(Ptr, getStaticErrorLLVMType(), getStaticErrorAlignment());
}

Address CodeGenFunction::createStaticErrorSlot(const Twine &Name) {
  if (!getContext().getStdErrorDecl())
    return makeStaticErrorAddress(
        CreateTempAlloca(llvm::ArrayType::get(Int8PtrTy, 2), Name));
  return CreateMemTemp(getContext().getStdErrorType(), Name);
}

llvm::Value *CodeGenFunction::loadStaticErrorDomain(Address Slot) {
  // The domain pointer is the first member of std::error (checked by Sema).
  return Builder.CreateLoad(Slot.withElementType(Int8PtrTy),
                            "static.error.domain");
}

void CodeGenFunction::clearStaticError(Address Slot) {
  Builder.CreateStore(llvm::ConstantPointerNull::get(Int8PtrTy),
                      Slot.withElementType(Int8PtrTy));
}

bool CodeGenFunction::isStaticErrorTriviallyDestructible() {
  return !getContext().getStdErrorDecl() ||
         !getContext().getStdErrorType().isDestructedType();
}

/// Construct the std::error at \p Dest from the one at \p Src with \p Ctor,
/// or copy the bits if std::error is trivially copyable (\p Ctor is null).
static void emitStaticErrorConstruct(CodeGenFunction &CGF,
                                     const CXXConstructorDecl *Ctor,
                                     Address Dest, Address Src) {
  QualType ErrTy = CGF.getContext().getStdErrorType();
  if (!Ctor) {
    CGF.EmitAggregateCopy(CGF.MakeAddrLValue(Dest, ErrTy),
                          CGF.MakeAddrLValue(Src, ErrTy), ErrTy,
                          AggValueSlot::MayOverlap);
    return;
  }
  CallArgList Args;
  Args.add(RValue::get(CGF.getAsNaturalPointerTo(Dest, ErrTy)),
           Ctor->getThisType());
  Args.add(RValue::get(CGF.getAsNaturalPointerTo(Src, ErrTy)),
           Ctor->getParamDecl(0)->getType());
  CGF.EmitCXXConstructorCall(Ctor, Ctor_Complete, /*ForVirtualBase=*/false,
                             /*Delegating=*/false, Dest, Args,
                             AggValueSlot::DoesNotOverlap, SourceLocation(),
                             /*NewPointerIsChecked=*/true);
}

void CodeGenFunction::EmitStaticErrorCopy(Address Dest, Address Src) {
  emitStaticErrorConstruct(
      *this, getContext().getStdErrorCopyConstructorDecl(), Dest, Src);
}

void CodeGenFunction::EmitStaticErrorMove(Address Dest, Address Src) {
  emitStaticErrorConstruct(
      *this, getContext().getStdErrorMoveConstructorDecl(), Dest, Src);
}

void CodeGenFunction::EmitStaticErrorRelocate(Address Dest, Address Src,
                                              bool ClearSource) {
  QualType ErrTy = getContext().getStdErrorType();
  Builder.CreateMemCpy(
      Dest, Src,
      llvm::ConstantInt::get(IntPtrTy,
                             getContext().getTypeSizeInChars(ErrTy).getQuantity()));
  if (ClearSource && !isStaticErrorTriviallyDestructible())
    clearStaticError(Src);
}

EHScopeStack::stable_iterator
CodeGenFunction::pushStaticErrorDestroy(Address Addr) {
  if (isStaticErrorTriviallyDestructible())
    return EHScopeStack::stable_end();
  QualType ErrTy = getContext().getStdErrorType();
  pushDestroy(NormalAndEHCleanup, Addr, ErrTy, destroyCXXObject,
              /*useEHCleanupForArray=*/false);
  return EHStack.stable_begin();
}

//===----------------------------------------------------------------------===//
// Returning errors in registers
//===----------------------------------------------------------------------===//

StaticErrorReturnLayout
CodeGen::computeStaticErrorReturnLayout(llvm::Type *ValueTy,
                                        const llvm::DataLayout &DL) {
  StaticErrorReturnLayout L;
  L.ValueType = ValueTy;
  llvm::LLVMContext &Ctx = ValueTy->getContext();
  unsigned WordBits = DL.getPointerSizeInBits();
  llvm::Type *WordTy = llvm::IntegerType::get(Ctx, WordBits);

  // Split the value into its scalar parts.
  SmallVector<unsigned, 2> Path;
  auto Flatten = [&](auto &Self, llvm::Type *Ty) -> void {
    if (auto *ST = dyn_cast<llvm::StructType>(Ty)) {
      for (unsigned I = 0, E = ST->getNumElements(); I != E; ++I) {
        Path.push_back(I);
        Self(Self, ST->getElementType(I));
        Path.pop_back();
      }
      return;
    }
    if (auto *AT = dyn_cast<llvm::ArrayType>(Ty)) {
      for (unsigned I = 0, E = AT->getNumElements(); I != E; ++I) {
        Path.push_back(I);
        Self(Self, AT->getElementType());
        Path.pop_back();
      }
      return;
    }
    StaticErrorReturnLayout::Part P;
    P.Path = Path;
    P.Ty = Ty;
    P.Field = 0;
    P.NumWords = 0;
    L.Parts.push_back(P);
  };
  if (!ValueTy->isVoidTy())
    Flatten(Flatten, ValueTy);

  // Integers and pointers go into words, everything else into fields of
  // their own. Pointers in other address spaces are not converted to and from
  // words, as the std::error's domain pointer is in the default one.
  SmallVector<llvm::Type *, 4> Words, Others;
  for (StaticErrorReturnLayout::Part &P : L.Parts) {
    bool IsWordPtr = P.Ty->isPointerTy() &&
                     P.Ty->getPointerAddressSpace() == 0 &&
                     DL.getTypeSizeInBits(P.Ty) == WordBits;
    if (!IsWordPtr && !P.Ty->isIntegerTy()) {
      P.Field = Others.size();
      Others.push_back(P.Ty);
      continue;
    }
    P.Field = Words.size();
    P.NumWords =
        IsWordPtr ? 1 : llvm::divideCeil(P.Ty->getIntegerBitWidth(), WordBits);
    if (IsWordPtr || P.Ty == WordTy)
      Words.push_back(P.Ty);
    else
      Words.append(P.NumWords, WordTy);
  }
  // Room for the std::error.
  while (Words.size() < 2)
    Words.push_back(WordTy);
  for (StaticErrorReturnLayout::Part &P : L.Parts)
    if (!P.NumWords)
      P.Field += Words.size();

  SmallVector<llvm::Type *, 8> Fields(Words.begin(), Words.end());
  Fields.append(Others);
  L.FlagField = Fields.size();
  Fields.push_back(llvm::Type::getInt1Ty(Ctx));
  L.Type = llvm::StructType::get(Ctx, Fields);
  return L;
}

/// Convert \p V (an integer of at most word size or a pointer) to the word
/// type \p WordTy.
static llvm::Value *toWord(CGBuilderTy &B, llvm::Value *V, llvm::Type *WordTy) {
  if (V->getType() == WordTy)
    return V;
  if (WordTy->isPointerTy())
    return B.CreateIntToPtr(V, WordTy);
  if (V->getType()->isPointerTy())
    return B.CreatePtrToInt(V, WordTy);
  return B.CreateZExt(V, WordTy);
}

/// The inverse of toWord.
static llvm::Value *fromWord(CGBuilderTy &B, llvm::Value *W, llvm::Type *Ty) {
  if (W->getType() == Ty)
    return W;
  if (Ty->isPointerTy())
    return B.CreateIntToPtr(W, Ty);
  if (W->getType()->isPointerTy())
    return B.CreatePtrToInt(W, Ty);
  return B.CreateTrunc(W, Ty);
}

/// The address of the value member of the std::error in \p Slot, which
/// follows the domain pointer.
static Address getStaticErrorValueAddress(CodeGenFunction &CGF, Address Slot) {
  return CGF.Builder
      .CreateConstInBoundsByteGEP(Slot.withElementType(CGF.Int8Ty),
                                  CGF.getPointerSize())
      .withElementType(CGF.IntPtrTy);
}

/// Check that std::error is two words, as the register ABI assumes.
static bool checkStaticErrorLayout(CodeGenFunction &CGF) {
  ASTContext &Ctx = CGF.getContext();
  if (Ctx.getStdErrorDecl() &&
      Ctx.getTypeSizeInChars(Ctx.getStdErrorType()) == 2 * CGF.getPointerSize())
    return true;
  CGF.CGM.ErrorUnsupported(CGF.CurFuncDecl,
                           "returning a std::error that is not two words in "
                           "registers");
  return false;
}

llvm::Value *
CodeGenFunction::EmitStaticSuccessReturnValue(const CGFunctionInfo &FI,
                                              llvm::Value *RV) {
  const StaticErrorReturnLayout &L =
      CGM.getTypes().getStaticErrorReturnLayout(FI);
  llvm::Value *Ret = llvm::PoisonValue::get(L.Type);
  if (RV) {
    assert(RV->getType() == L.ValueType && "unexpected return value");
    for (const StaticErrorReturnLayout::Part &P : L.Parts) {
      llvm::Value *V =
          P.Path.empty() ? RV : Builder.CreateExtractValue(RV, P.Path);
      if (!P.NumWords) {
        Ret = Builder.CreateInsertValue(Ret, V, P.Field);
        continue;
      }
      if (P.NumWords == 1) {
        Ret = Builder.CreateInsertValue(
            Ret, toWord(Builder, V, L.Type->getElementType(P.Field)), P.Field);
        continue;
      }
      // An integer that is wider than a word.
      llvm::Type *WideTy = llvm::IntegerType::get(
          getLLVMContext(), P.NumWords * IntPtrTy->getBitWidth());
      V = Builder.CreateZExt(V, WideTy);
      for (unsigned I = 0; I != P.NumWords; ++I)
        Ret = Builder.CreateInsertValue(
            Ret,
            Builder.CreateTrunc(
                Builder.CreateLShr(V, I * IntPtrTy->getBitWidth()), IntPtrTy),
            P.Field + I);
    }
  }
  return Builder.CreateInsertValue(Ret, Builder.getFalse(), L.FlagField);
}

/// Return the error in the current function's slot.
static void emitStaticErrorReturn(CodeGenFunction &CGF) {
  CGBuilderTy &B = CGF.Builder;
  const StaticErrorReturnLayout &L =
      CGF.CGM.getTypes().getStaticErrorReturnLayout(*CGF.CurFnInfo);
  llvm::Value *Ret = llvm::PoisonValue::get(L.Type);
  if (checkStaticErrorLayout(CGF)) {
    Address Slot = CGF.StaticErrorOutSlot;
    llvm::Value *Domain = CGF.loadStaticErrorDomain(Slot);
    llvm::Value *Value = B.CreateLoad(getStaticErrorValueAddress(CGF, Slot),
                                      "static.error.value");
    Ret = B.CreateInsertValue(Ret, toWord(B, Domain, L.Type->getElementType(0)),
                              0);
    Ret = B.CreateInsertValue(Ret, toWord(B, Value, L.Type->getElementType(1)),
                              1);
  }
  B.CreateRet(B.CreateInsertValue(Ret, B.getTrue(), L.FlagField));
}

llvm::Value *CodeGenFunction::EmitStaticErrorCallResult(
    const CGFunctionInfo &FI, llvm::Value *Result, const StaticErrorTarget &T) {
  const StaticErrorReturnLayout &L =
      CGM.getTypes().getStaticErrorReturnLayout(FI);
  if (!HaveInsertPoint())
    return L.ValueType->isVoidTy() ? nullptr
                                   : llvm::PoisonValue::get(L.ValueType);

  llvm::Value *Failed =
      Builder.CreateExtractValue(Result, L.FlagField, "static.failed");
  llvm::BasicBlock *ErrorBB = createBasicBlock("static.unwind");
  llvm::BasicBlock *ContBB = createBasicBlock("static.cont");
  // Errors are exceptional.
  Builder.CreateCondBr(
      Failed, ErrorBB, ContBB,
      llvm::MDBuilder(getLLVMContext()).createUnlikelyBranchWeights());

  EmitBlock(ErrorBB);
  if (checkStaticErrorLayout(*this)) {
    Builder.CreateStore(
        fromWord(Builder, Builder.CreateExtractValue(Result, 0), Int8PtrTy),
        T.Slot.withElementType(Int8PtrTy));
    Builder.CreateStore(
        fromWord(Builder, Builder.CreateExtractValue(Result, 1), IntPtrTy),
        getStaticErrorValueAddress(*this, T.Slot));
  }
  EmitStaticErrorExit(T);
  EmitBlock(ContBB);

  if (L.ValueType->isVoidTy())
    return nullptr;
  llvm::Value *V = llvm::PoisonValue::get(L.ValueType);
  for (const StaticErrorReturnLayout::Part &P : L.Parts) {
    llvm::Value *PartV;
    if (!P.NumWords) {
      PartV = Builder.CreateExtractValue(Result, P.Field);
    } else if (P.NumWords == 1) {
      PartV =
          fromWord(Builder, Builder.CreateExtractValue(Result, P.Field), P.Ty);
    } else {
      // An integer that is wider than a word.
      unsigned WordBits = IntPtrTy->getBitWidth();
      llvm::Type *WideTy =
          llvm::IntegerType::get(getLLVMContext(), P.NumWords * WordBits);
      PartV = llvm::ConstantInt::get(WideTy, 0);
      for (unsigned I = 0; I != P.NumWords; ++I)
        PartV = Builder.CreateOr(
            PartV,
            Builder.CreateShl(
                Builder.CreateZExt(
                    Builder.CreateExtractValue(Result, P.Field + I), WideTy),
                I * WordBits));
      PartV = Builder.CreateTrunc(PartV, P.Ty);
    }
    if (P.Path.empty())
      return PartV;
    V = Builder.CreateInsertValue(V, PartV, P.Path);
  }
  return V;
}

bool CodeGenFunction::canForwardStaticErrorResult(
    const CGFunctionInfo &CalleeInfo, llvm::Type *CalleeRetTy) {
  if (!isStaticThrowsFunction() || !CalleeInfo.hasStaticErrorParam() ||
      !StaticErrorOutSlot.isValid())
    return false;
  // The error must go to our caller, without calling the propagation hook.
  if (!StaticErrorHandlers.empty() ||
      getLangOpts().StaticExceptionsPropagationHook ||
      ShouldInstrumentFunction())
    return false;
  // The result must be returned the same way (the caller checked that the
  // types are the same).
  if (CurFn->getReturnType() != CalleeRetTy ||
      CurFnInfo->getReturnInfo().getKind() !=
          CalleeInfo.getReturnInfo().getKind() ||
      CurFnInfo->getEffectiveCallingConvention() !=
          CalleeInfo.getEffectiveCallingConvention())
    return false;
  // No cleanups may run between the call and the return.
  for (EHScopeStack::iterator I = EHStack.begin(),
                              E = EHStack.find(PrologueCleanupDepth);
       I != E; ++I) {
    auto *Scope = dyn_cast<EHCleanupScope>(&*I);
    if (Scope && (Scope->isActive() || Scope->getActiveFlag().isValid()))
      return false;
  }
  return true;
}

llvm::BasicBlock *CodeGenFunction::getStaticErrorReturnBlock() {
  FunctionDecl *Hook = getContext().getStdNotifyErrorPropagationDecl();
  // A thunk only forwards the error of the function it calls, which already
  // called the hook.
  bool UseHook =
      Hook && getLangOpts().StaticExceptionsPropagationHook && !CurFuncIsThunk;
  bool InRegisters = CGM.getTypes().returnsStaticErrorInRegisters(*CurFnInfo);
  if (!UseHook && !InRegisters)
    return ReturnBlock.getBlock();
  if (StaticErrorReturnBlock)
    return StaticErrorReturnBlock;

  // One shared exit that calls the hook with the error (P0709 4.4) and/or
  // returns the error in registers.
  StaticErrorReturnBlock = createBasicBlock("static.error.return");
  CGBuilderTy::InsertPoint SavedIP = Builder.saveAndClearIP();
  CurFn->insert(CurFn->end(), StaticErrorReturnBlock);
  Builder.SetInsertPoint(StaticErrorReturnBlock);
  if (UseHook) {
    CallArgList Args;
    Args.add(RValue::get(StaticErrorOutSlot, *this),
             getContext().getLValueReferenceType(
                 getContext().getStdErrorType().withConst()));
    const CGFunctionInfo &FnInfo =
        CGM.getTypes().arrangeFunctionDeclaration(Hook);
    llvm::Constant *Fn = CGM.GetAddrOfFunction(Hook);
    llvm::CallBase *Call;
    EmitCall(FnInfo, CGCallee::forDirect(Fn, GlobalDecl(Hook)),
             ReturnValueSlot(), Args, &Call);
    Call->setDoesNotThrow();
  }
  if (InRegisters)
    emitStaticErrorReturn(*this);
  else
    Builder.CreateBr(ReturnBlock.getBlock());
  Builder.restoreIP(SavedIP);
  return StaticErrorReturnBlock;
}

CodeGenFunction::StaticErrorTarget CodeGenFunction::getStaticErrorTarget() {
  if (!StaticErrorHandlers.empty()) {
    StaticErrorTarget &T = StaticErrorHandlers.back();
    if (T.Kind == StaticErrorTarget::Terminate && !T.Slot.isValid()) {
      if (!StaticErrorTempSlot.isValid())
        StaticErrorTempSlot = createStaticErrorSlot("static.error.tmp");
      T.Slot = StaticErrorTempSlot;
    }
    return T;
  }

  StaticErrorTarget T;
  T.Try = nullptr;
  T.HandlerIndex = 0;
  T.Depth = PrologueCleanupDepth;
  if (isStaticThrowsFunction() && StaticErrorOutSlot.isValid()) {
    T.Kind = StaticErrorTarget::ReturnToCaller;
    T.Slot = StaticErrorOutSlot;
    T.Block = getStaticErrorReturnBlock();
    return T;
  }

  if (!StaticErrorTempSlot.isValid())
    StaticErrorTempSlot = createStaticErrorSlot("static.error.tmp");
  T.Kind = StaticErrorTarget::DynamicThrow;
  T.Slot = StaticErrorTempSlot;
  T.Block = nullptr;
  return T;
}

void CodeGenFunction::EmitStaticErrorCheck(const StaticErrorTarget &T) {
  if (!HaveInsertPoint())
    return;

  llvm::Value *Failed =
      Builder.CreateIsNotNull(loadStaticErrorDomain(T.Slot), "static.failed");
  llvm::BasicBlock *ErrorBB = createBasicBlock("static.unwind");
  llvm::BasicBlock *ContBB = createBasicBlock("static.cont");
  // Errors are exceptional.
  Builder.CreateCondBr(
      Failed, ErrorBB, ContBB,
      llvm::MDBuilder(getLLVMContext()).createUnlikelyBranchWeights());

  EmitBlock(ErrorBB);
  EmitStaticErrorExit(T);
  EmitBlock(ContBB);
}

void CodeGenFunction::EmitStaticErrorExit(const StaticErrorTarget &T) {
  if (!HaveInsertPoint())
    return;

  switch (T.Kind) {
  case StaticErrorTarget::ReturnToCaller:
  case StaticErrorTarget::LocalHandler:
    // A static exception unwinds exactly like a dynamic one, but by a forward
    // branch: run all the cleanups between here and the target inline.
    EmitStaticErrorCleanups(T.Depth);
    Builder.CreateBr(T.Block);
    Builder.ClearInsertionPoint();
    return;

  case StaticErrorTarget::Terminate:
    EmitNounwindRuntimeCall(CGM.getTerminateFn())->setDoesNotReturn();
    Builder.CreateUnreachable();
    Builder.ClearInsertionPoint();
    return;

  case StaticErrorTarget::DynamicThrow:
    EmitThrowStaticErrorAsDynamic(T.Slot);
    // EmitCall leaves a dummy insertion point after a noreturn call.
    if (HaveInsertPoint()) {
      Builder.CreateUnreachable();
      Builder.ClearInsertionPoint();
    }
    return;
  }
  llvm_unreachable("invalid static error target");
}

void CodeGenFunction::EmitThrowStaticErrorAsDynamic(Address Slot) {
  FunctionDecl *FD = getContext().getStdThrowErrorAsDynamicDecl();
  if (!FD || !getContext().getStdErrorDecl()) {
    CGM.ErrorUnsupported(CurFuncDecl, "static exception without "
                                      "std::__throw_error_as_dynamic");
    EmitTrapCall(llvm::Intrinsic::trap);
    return;
  }

  // Move the error out of the slot before the call: the slot might be reused
  // after the dynamic exception is caught in this function. The callee
  // destroys its by-value parameter.
  QualType ErrTy = getContext().getStdErrorType();
  Address Tmp = CreateMemTemp(ErrTy, "static.error.arg");
  EmitStaticErrorRelocate(Tmp, Slot, /*ClearSource=*/false);
  clearStaticError(Slot);

  CallArgList Args;
  Args.add(RValue::getAggregate(Tmp), ErrTy);
  const CGFunctionInfo &FnInfo = CGM.getTypes().arrangeFunctionDeclaration(FD);
  llvm::Constant *Fn = CGM.GetAddrOfFunction(FD);
  EmitCall(FnInfo, CGCallee::forDirect(Fn, GlobalDecl(FD)), ReturnValueSlot(),
           Args);
}

void CodeGenFunction::EmitStaticErrorCleanups(
    EHScopeStack::stable_iterator Depth) {
  EmitStaticErrorCleanups(EHStack.stable_begin(), Depth);
}

void CodeGenFunction::EmitStaticErrorCleanups(
    EHScopeStack::stable_iterator From, EHScopeStack::stable_iterator To) {
  // Collect the cleanups first: emitting them may push (and pop) scopes,
  // which invalidates iterators, but not stable iterators.
  SmallVector<EHScopeStack::stable_iterator, 8> Cleanups;
  for (EHScopeStack::iterator I = EHStack.find(From), E = EHStack.find(To);
       I != E; ++I) {
    // Catch, filter and terminate scopes only affect dynamic exceptions.
    // Cleanups with an activation flag are tested at run time.
    auto *Scope = dyn_cast<EHCleanupScope>(&*I);
    if (Scope && (Scope->isActive() || Scope->getActiveFlag().isValid()))
      Cleanups.push_back(EHStack.stabilize(I));
  }
  if (Cleanups.empty())
    return;

  // Like for dynamic exceptions, an exception escaping a cleanup while a
  // static exception is being propagated terminates the program.
  bool PushedTerminate = false;
  if (CGM.getLangOpts().Exceptions) {
    EHStack.pushTerminate();
    PushedTerminate = true;
  }

  // The cleanups stay on the stack while they are emitted, so a static
  // exception escaping one of them must not propagate through them again.
  pushStaticErrorTerminate();
  for (EHScopeStack::stable_iterator SI : Cleanups)
    EmitCleanupInline(SI);
  popStaticErrorTerminate();

  if (PushedTerminate)
    EHStack.popTerminate();
}

void CodeGenFunction::pushStaticErrorTerminate() {
  StaticErrorTarget T;
  T.Kind = StaticErrorTarget::Terminate;
  T.Depth = EHStack.stable_begin();
  StaticErrorHandlers.push_back(T);
}

void CodeGenFunction::popStaticErrorTerminate() {
  assert(!StaticErrorHandlers.empty() &&
         StaticErrorHandlers.back().Kind == StaticErrorTarget::Terminate &&
         "unbalanced static error terminate scope");
  StaticErrorHandlers.pop_back();
}

bool CodeGenFunction::isStaticThrowExpr(const CXXThrowExpr *E) {
  // Sema converts the operand of a throw-expression in a 'throws' function
  // to std::error if possible; everything else is a dynamic exception.
  const Expr *SubExpr = E->getSubExpr();
  return SubExpr && isStaticThrowsFunction() &&
         getContext().isStdErrorType(SubExpr->getType());
}

void CodeGenFunction::EmitStaticThrowExpr(const CXXThrowExpr *E) {
  StaticErrorTarget T = getStaticErrorTarget();
  const Expr *SubExpr = E->getSubExpr();
  // Evaluate the operand directly into the target's slot. If a nested call
  // to a 'throws' function fails, it overwrites the slot and branches to the
  // same target, which is what we want.
  Address Slot = T.Slot.withElementType(ConvertTypeForMem(SubExpr->getType()));
  EmitAnyExprToMem(SubExpr, Slot, SubExpr->getType().getQualifiers(),
                   /*IsInit=*/true);
  EmitStaticErrorExit(T);
}

void CodeGenFunction::EmitStaticErrorMoveExit(
    const StaticErrorTarget &T, Address Src,
    EHScopeStack::stable_iterator SrcCleanup) {
  if (!HaveInsertPoint())
    return;

  if (SrcCleanup == EHScopeStack::stable_end()) {
    // Trivially destructible: a copy of the bits is a move.
    EmitStaticErrorRelocate(T.Slot, Src, /*ClearSource=*/false);
    EmitStaticErrorExit(T);
    return;
  }

  switch (T.Kind) {
  case StaticErrorTarget::ReturnToCaller:
  case StaticErrorTarget::LocalHandler: {
    if (!T.Depth.strictlyEncloses(SrcCleanup)) {
      // The target is inside the scope of Src (a try statement in the
      // handler that owns Src), so Src outlives the exit and is destroyed
      // later: move from it, leaving a valid error.
      EmitStaticErrorMove(T.Slot, Src);
      EmitStaticErrorExit(T);
      return;
    }
    // Run the cleanups inside the scope of Src, then move Src to the target
    // instead of destroying it, then run the remaining cleanups.
    EmitStaticErrorCleanups(EHStack.stable_begin(), SrcCleanup);
    EmitStaticErrorRelocate(T.Slot, Src, /*ClearSource=*/false);
    EHScopeStack::iterator BelowI = EHStack.find(SrcCleanup);
    ++BelowI;
    EHScopeStack::stable_iterator Below = EHStack.stabilize(BelowI);
    EmitStaticErrorCleanups(Below, T.Depth);
    Builder.CreateBr(T.Block);
    Builder.ClearInsertionPoint();
    return;
  }

  case StaticErrorTarget::DynamicThrow:
  case StaticErrorTarget::Terminate:
    // The cleanups run during unwinding and destroy Src, so the exception
    // gets a copy (which shares a wrapped dynamic exception).
    EmitStaticErrorCopy(T.Slot, Src);
    EmitStaticErrorExit(T);
    return;
  }
  llvm_unreachable("invalid static error target");
}

bool CodeGenFunction::EmitStaticRethrow() {
  if (StaticCatchStack.empty())
    return false;
  const StaticCatchInfo &Info = StaticCatchStack.back();
  if (Info.Kind == StaticCatchInfo::CK_DynamicOnly)
    return false;

  StaticErrorTarget T = getStaticErrorTarget();

  if (Info.Kind == StaticCatchInfo::CK_ErrorVar) {
    // P0709: in a handler for std::error, 'throw;' rethrows the caught error
    // (the catch parameter, or the slot it refers to).
    Address Src = Info.ErrorVar->getType()->isReferenceType()
                      ? Info.Slot
                      : GetAddrOfLocalVar(Info.ErrorVar);
    EmitStaticErrorMoveExit(T, Src, Info.ErrorCleanup);
    return true;
  }

  // catch(...): rethrow whatever kind of exception we caught.
  assert(Info.Kind == StaticCatchInfo::CK_CatchAll);
  if (!Info.DynamicFlag.isValid()) {
    // Only static exceptions can reach this handler.
    EmitStaticErrorMoveExit(T, Info.Slot, Info.ErrorCleanup);
    return true;
  }

  llvm::BasicBlock *DynamicBB = createBasicBlock("rethrow.dynamic");
  llvm::BasicBlock *StaticBB = createBasicBlock("rethrow.static");
  Builder.CreateCondBr(Builder.CreateLoad(Info.DynamicFlag), DynamicBB,
                       StaticBB);
  EmitBlock(DynamicBB);
  CGM.getCXXABI().emitRethrow(*this, /*isNoReturn=*/true);
  EmitBlock(StaticBB);
  EmitStaticErrorMoveExit(T, Info.Slot, Info.ErrorCleanup);
  return true;
}

static llvm::FunctionCallee getBeginCatchFn(CodeGenModule &CGM) {
  // void *__cxa_begin_catch(void*);
  llvm::FunctionType *FTy =
      llvm::FunctionType::get(CGM.Int8PtrTy, CGM.Int8PtrTy, /*isVarArg=*/false);
  return CGM.CreateRuntimeFunction(FTy, "__cxa_begin_catch");
}

static llvm::FunctionCallee getEndCatchFn(CodeGenModule &CGM) {
  // void __cxa_end_catch();
  llvm::FunctionType *FTy =
      llvm::FunctionType::get(CGM.VoidTy, /*isVarArg=*/false);
  return CGM.CreateRuntimeFunction(FTy, "__cxa_end_catch");
}

EHScopeStack::stable_iterator
CodeGenFunction::EnterStaticExceptionTranslation() {
  EHCatchScope *CatchScope = EHStack.pushCatch(1);
  CatchScope->setHandler(0, CGM.getCXXABI().getCatchAllTypeInfo(),
                         createBasicBlock("static.translate"));
  return EHStack.stable_begin();
}

void CodeGenFunction::ExitStaticExceptionTranslation(
    EHScopeStack::stable_iterator Depth) {
  // The cleanups of the function body's outermost scope are still active;
  // they must be inside the translation scope.
  PopCleanupBlocks(Depth);

  EHCatchScope &CatchScope = cast<EHCatchScope>(*EHStack.begin());
  if (!CatchScope.hasEHBranches()) {
    CatchScope.clearHandlerBlocks();
    EHStack.popCatch();
    return;
  }
  // A single catch-all is its own dispatch block.
  llvm::BasicBlock *Handler = CatchScope.getHandler(0).Block;
  EHStack.popCatch();

  CGBuilderTy::InsertPoint SavedIP = Builder.saveAndClearIP();
  EmitBlockAfterUses(Handler);

  // catch (...) { error e = std::__error_from_current_exception(); return e; }
  llvm::Value *Exn = getExceptionFromSlot();
  EmitNounwindRuntimeCall(getBeginCatchFn(CGM), Exn);

  FunctionDecl *Translate = getContext().getStdErrorFromCurrentExceptionDecl();
  if (Translate && StaticErrorOutSlot.isValid()) {
    const CGFunctionInfo &FnInfo =
        CGM.getTypes().arrangeFunctionDeclaration(Translate);
    llvm::Constant *Fn = CGM.GetAddrOfFunction(Translate);
    EHStack.pushTerminate();
    EmitCall(FnInfo, CGCallee::forDirect(Fn, GlobalDecl(Translate)),
             ReturnValueSlot(StaticErrorOutSlot, /*IsVolatile=*/false),
             CallArgList());
    // Destroying the exception object might throw; that terminates.
    EmitRuntimeCallOrInvoke(getEndCatchFn(CGM));
    EHStack.popTerminate();
  } else {
    CGM.ErrorUnsupported(CurFuncDecl, "'throws' without "
                                      "std::__error_from_current_exception");
    EmitTrapCall(llvm::Intrinsic::trap);
  }

  StaticErrorTarget T;
  T.Kind = StaticErrorTarget::ReturnToCaller;
  T.Slot = StaticErrorOutSlot;
  T.Block = getStaticErrorReturnBlock();
  T.Depth = PrologueCleanupDepth;
  T.Try = nullptr;
  T.HandlerIndex = 0;
  EmitStaticErrorExit(T);

  Builder.restoreIP(SavedIP);
}

namespace {
/// At the end of a catch(...) handler, end the catch if it was entered by a
/// dynamic exception, and destroy the static exception otherwise.
struct EndCatchOrDestroyStaticError final : EHScopeStack::Cleanup {
  Address DynamicFlag;
  Address Slot;
  EndCatchOrDestroyStaticError(Address DynamicFlag, Address Slot)
      : DynamicFlag(DynamicFlag), Slot(Slot) {}

  void Emit(CodeGenFunction &CGF, Flags flags) override {
    llvm::BasicBlock *EndBB = CGF.createBasicBlock("catch.end.dynamic");
    llvm::BasicBlock *StaticBB = CGF.createBasicBlock("catch.end.static");
    llvm::BasicBlock *DoneBB = CGF.createBasicBlock("catch.end.done");
    bool Destroy = !CGF.isStaticErrorTriviallyDestructible();
    CGF.Builder.CreateCondBr(CGF.Builder.CreateLoad(DynamicFlag), EndBB,
                             Destroy ? StaticBB : DoneBB);
    CGF.EmitBlock(EndBB);
    // Destroying the exception object might throw.
    CGF.EmitRuntimeCallOrInvoke(getEndCatchFn(CGF.CGM));
    CGF.Builder.CreateBr(DoneBB);
    if (Destroy) {
      CGF.EmitBlock(StaticBB);
      QualType ErrTy = CGF.getContext().getStdErrorType();
      CGF.destroyCXXObject(CGF, Slot, ErrTy);
    } else {
      delete StaticBB;
    }
    CGF.EmitBlock(DoneBB);
  }
};
} // namespace

void CodeGenFunction::EmitStaticCatchHandler(const CXXCatchStmt *C,
                                             llvm::BasicBlock *DynamicEntry,
                                             const StaticErrorTarget &Target,
                                             llvm::BasicBlock *ContBB,
                                             bool ImplicitRethrow) {
  Address Slot = Target.Slot;
  bool IsCatchAll = !C->getExceptionDecl();
  llvm::BasicBlock *BodyBB = createBasicBlock("catch.body");

  // Dynamic entry: the handler matched a dynamic exception.
  Address DynamicFlag = Address::invalid();
  if (DynamicEntry) {
    if (IsCatchAll)
      DynamicFlag = CreateTempAlloca(Builder.getInt1Ty(), CharUnits::One(),
                                     "catch.is.dynamic");
    EmitBlockAfterUses(DynamicEntry);
    llvm::Value *Exn = getExceptionFromSlot();
    llvm::Value *Obj = EmitNounwindRuntimeCall(getBeginCatchFn(CGM), Exn);
    if (IsCatchAll) {
      Builder.CreateStore(Builder.getTrue(), DynamicFlag);
    } else {
      // A std::error caught as a dynamic exception: copy it out and finish
      // the dynamic exception right away, so both entries share the same
      // handler body.
      EmitStaticErrorCopy(Slot, makeStaticErrorAddress(Obj));
      EmitNounwindRuntimeCall(getEndCatchFn(CGM));
    }
    Builder.CreateBr(BodyBB);
  }

  // Static entry: a static exception was delivered into Slot.
  EmitBlock(Target.Block);
  if (DynamicFlag.isValid())
    Builder.CreateStore(Builder.getFalse(), DynamicFlag);
  EmitBlock(BodyBB);

  RunCleanupsScope CatchScope(*this);
  StaticCatchInfo Info;
  Info.Slot = Slot;
  Info.DynamicFlag = DynamicFlag;
  Info.ErrorVar = nullptr;
  if (IsCatchAll) {
    Info.Kind = StaticCatchInfo::CK_CatchAll;
    if (DynamicFlag.isValid()) {
      EHStack.pushCleanup<EndCatchOrDestroyStaticError>(NormalAndEHCleanup,
                                                        DynamicFlag, Slot);
      if (!isStaticErrorTriviallyDestructible())
        Info.ErrorCleanup = EHStack.stable_begin();
    } else {
      Info.ErrorCleanup = pushStaticErrorDestroy(Slot);
    }
  } else {
    // Initialize the catch parameter from the slot.
    const VarDecl *Var = C->getExceptionDecl();
    AutoVarEmission Emission = EmitAutoVarAlloca(*Var);
    Address VarAddr = Emission.getAllocatedAddress();
    if (Var->getType()->isReferenceType()) {
      Builder.CreateStore(Slot.emitRawPointer(*this), VarAddr);
      EmitAutoVarCleanups(Emission);
      // The reference refers to the slot, which the handler owns.
      Info.ErrorCleanup = pushStaticErrorDestroy(Slot);
    } else {
      // The catch parameter takes over the error.
      EmitStaticErrorRelocate(VarAddr.withElementType(Slot.getElementType()),
                              Slot, /*ClearSource=*/false);
      EHScopeStack::stable_iterator Before = EHStack.stable_begin();
      EmitAutoVarCleanups(Emission);
      if (EHStack.stable_begin() != Before)
        Info.ErrorCleanup = EHStack.stable_begin();
    }
    Info.Kind = StaticCatchInfo::CK_ErrorVar;
    Info.ErrorVar = Var;
  }

  StaticCatchStack.push_back(Info);
  incrementProfileCounter(C);
  EmitStmt(C->getHandlerBlock());

  // [except.handle]p11: falling off the end of a handler of the
  // function-try-block of a constructor or destructor rethrows.
  if (ImplicitRethrow && HaveInsertPoint())
    EmitStaticRethrow();
  StaticCatchStack.pop_back();

  CatchScope.ForceCleanup();
  if (HaveInsertPoint())
    Builder.CreateBr(ContBB);
}
