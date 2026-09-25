//===--- SemaStaticException.cpp - P0709 static exceptions ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Semantic analysis for the prototype of P0709 "Zero-overhead deterministic
// exceptions: Throwing values" (enabled with -fstatic-exceptions).
//
// A function declared with the static-exception-specification 'throws'
// reports failure by "throwing" a value of type std::error, which is
// returned to the caller through a hidden out-parameter (see
// CGStaticException.cpp). The library parts live in the <error> header
// shipped with clang:
//
//   std::error                              - the error type
//   std::__error_from_current_exception()   - dynamic -> static translation
//   std::__throw_error_as_dynamic(error)    - static -> dynamic translation
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace clang;

static FunctionDecl *lookupStdHelper(Sema &S, NamespaceDecl *Std,
                                     StringRef Name, SourceLocation Loc) {
  LookupResult R(S, &S.getPreprocessor().getIdentifierTable().get(Name), Loc,
                 Sema::LookupOrdinaryName);
  S.LookupQualifiedName(R, Std);
  return R.getAsSingle<FunctionDecl>();
}

bool Sema::CheckStaticExceptionSupport(SourceLocation Loc) {
  // Only success is cached: std::error might be declared later.
  if (StaticExceptionSupport)
    return true;

  auto Fail = [&](unsigned DiagID, StringRef Arg = {}) {
    if (Arg.empty())
      Diag(Loc, DiagID);
    else
      Diag(Loc, DiagID) << Arg;
    return false;
  };

  if (!getLangOpts().CPlusPlus17)
    return Fail(diag::err_static_exceptions_requires_cxx17);

  // The implementation branches directly into catch handlers, which requires
  // landing-pad based EH (not funclets).
  const TargetInfo &TI = Context.getTargetInfo();
  if (TI.getCXXABI().isMicrosoft() || TI.getTriple().isWasm())
    return Fail(diag::err_static_exceptions_unsupported_abi);

  // Find std::error.
  NamespaceDecl *Std = getStdNamespace();
  CXXRecordDecl *ErrorRD = nullptr;
  if (Std) {
    LookupResult R(*this, &PP.getIdentifierTable().get("error"), Loc,
                   LookupTagName);
    LookupQualifiedName(R, Std);
    ErrorRD = R.getAsSingle<CXXRecordDecl>();
  }
  if (!ErrorRD || !ErrorRD->hasDefinition())
    return Fail(diag::err_static_exceptions_no_std_error);
  ErrorRD = ErrorRD->getDefinition();

  // The implementation moves errors with memcpy, so std::error must be
  // trivially relocatable: trivially copyable or [[clang::trivial_abi]]. It
  // uses the first member (the "domain") as the discriminant: it is never null
  // for a valid error, and the destructor must do nothing if it is null (the
  // state of a std::error that was moved from with memcpy).
  bool Valid = !ErrorRD->isInvalidDecl() && !ErrorRD->isDependentType() &&
               !ErrorRD->isPolymorphic() && ErrorRD->getNumBases() == 0 &&
               ErrorRD->canPassInRegisters() && !ErrorRD->field_empty();
  if (Valid) {
    const FieldDecl *First = *ErrorRD->field_begin();
    Valid = !First->isBitField() && First->getType()->isPointerType();
  }
  // Code generation copies and destroys errors (e.g. when a handler for
  // std::error is entered by a dynamic exception).
  CXXConstructorDecl *CopyCtor = nullptr;
  CXXConstructorDecl *MoveCtor = nullptr;
  CXXDestructorDecl *Dtor = nullptr;
  if (Valid && !ErrorRD->isTriviallyCopyable()) {
    CopyCtor = LookupCopyingConstructor(ErrorRD, Qualifiers::Const);
    MoveCtor = LookupMovingConstructor(ErrorRD, /*Quals=*/0);
    Dtor = LookupDestructor(ErrorRD);
    Valid = CopyCtor && !CopyCtor->isDeleted() &&
            CopyCtor->getNumParams() == 1 && MoveCtor &&
            !MoveCtor->isDeleted() && MoveCtor->getNumParams() == 1 && Dtor &&
            !Dtor->isDeleted();
  }
  if (!Valid) {
    Diag(ErrorRD->getLocation(), diag::err_static_exceptions_invalid_std_error);
    return false;
  }

  // Find the translation helpers.
  FunctionDecl *ThrowAsDynamic =
      lookupStdHelper(*this, Std, "__throw_error_as_dynamic", Loc);
  if (!ThrowAsDynamic || ThrowAsDynamic->getNumParams() != 1 ||
      !ThrowAsDynamic->getReturnType()->isVoidType() ||
      !Context.hasSameUnqualifiedType(
          ThrowAsDynamic->getParamDecl(0)->getType(),
          Context.getCanonicalTagType(ErrorRD)))
    return Fail(diag::err_static_exceptions_missing_helper,
                "__throw_error_as_dynamic");

  FunctionDecl *FromCurrent = nullptr;
  if (getLangOpts().CXXExceptions) {
    FromCurrent =
        lookupStdHelper(*this, Std, "__error_from_current_exception", Loc);
    if (!FromCurrent || FromCurrent->getNumParams() != 0 ||
        !Context.hasSameUnqualifiedType(FromCurrent->getReturnType(),
                                        Context.getCanonicalTagType(ErrorRD)))
      return Fail(diag::err_static_exceptions_missing_helper,
                  "__error_from_current_exception");
  }

  FunctionDecl *NotifyPropagation = nullptr;
  if (getLangOpts().StaticExceptionsPropagationHook) {
    NotifyPropagation =
        lookupStdHelper(*this, Std, "__notify_error_propagation", Loc);
    if (!NotifyPropagation || NotifyPropagation->getNumParams() != 1)
      return Fail(diag::err_static_exceptions_missing_helper,
                  "__notify_error_propagation");
  }

  Context.setStaticExceptionDecls(ErrorRD, FromCurrent, ThrowAsDynamic);
  Context.setStdNotifyErrorPropagationDecl(NotifyPropagation);
  Context.setStdErrorCopyConstructorDecl(CopyCtor);
  Context.setStdErrorMoveConstructorDecl(MoveCtor);
  StaticExceptionSupport = true;

  // Code generation calls these functions implicitly.
  MarkFunctionReferenced(Loc, ThrowAsDynamic);
  if (FromCurrent)
    MarkFunctionReferenced(Loc, FromCurrent);
  if (NotifyPropagation)
    MarkFunctionReferenced(Loc, NotifyPropagation);
  if (CopyCtor)
    MarkFunctionReferenced(Loc, CopyCtor);
  if (MoveCtor)
    MarkFunctionReferenced(Loc, MoveCtor);
  if (Dtor)
    MarkFunctionReferenced(Loc, Dtor);
  return true;
}

void Sema::CheckStaticExceptionFunctionDecl(FunctionDecl *FD) {
  const auto *FPT = FD->getType()->getAs<FunctionProtoType>();
  if (!FPT || (!FPT->hasStaticExceptionSpec() &&
               FPT->getExceptionSpecType() != EST_DependentThrows))
    return;

  // P0709: a conditional static exception specification is not allowed on
  // virtual functions.
  if (FPT->getExceptionSpecType() == EST_DependentThrows) {
    if (const auto *MD = dyn_cast<CXXMethodDecl>(FD); MD && MD->isVirtual())
      Diag(FD->getLocation(), diag::err_static_exception_cond_virtual)
          << FD->getExceptionSpecSourceRange();
    return;
  }

  int Kind = -1;
  if (isa<CXXDestructorDecl>(FD))
    Kind = 0;
  else if (FD->isMain())
    Kind = 1;
  else if (FD->getOverloadedOperator() == OO_Delete ||
           FD->getOverloadedOperator() == OO_Array_Delete)
    Kind = 4;
  if (Kind >= 0) {
    Diag(FD->getLocation(), diag::err_static_exception_spec_not_allowed)
        << Kind << FD->getExceptionSpecSourceRange();
    FD->setInvalidDecl();
  }
}

bool Sema::isInStaticExceptionFunction() const {
  // Blocks and captured statements are separate functions that cannot be
  // declared 'throws'; lambdas have their own call operator.
  const auto *FD = dyn_cast<FunctionDecl>(CurContext);
  if (!FD)
    return false;
  const auto *FPT = FD->getType()->getAs<FunctionProtoType>();
  return FPT && FPT->hasStaticExceptionSpec();
}

bool Sema::isInDependentStaticExceptionFunction() const {
  const auto *FD = dyn_cast<FunctionDecl>(CurContext);
  if (!FD)
    return false;
  const auto *FPT = FD->getType()->getAs<FunctionProtoType>();
  return FPT && FPT->getExceptionSpecType() == EST_DependentThrows;
}

ExprResult Sema::BuildStaticThrowOperand(SourceLocation ThrowLoc, Expr *E) {
  if (!Context.getStdErrorDecl() && !CheckStaticExceptionSupport(ThrowLoc))
    return ExprEmpty();

  QualType ErrTy = Context.getStdErrorType();
  InitializedEntity Entity =
      InitializedEntity::InitializeException(ThrowLoc, ErrTy);
  InitializationKind Kind =
      InitializationKind::CreateCopy(E->getBeginLoc(), ThrowLoc);
  InitializationSequence Seq(*this, Entity, Kind, E);
  if (!Seq)
    return ExprEmpty();
  return Seq.Perform(*this, Entity, Kind, E);
}

ExprResult
Sema::ActOnStaticExceptionSpecCondition(Expr *Cond, SourceLocation Loc,
                                        ExceptionSpecificationType &EST) {
  // A conditional 'throws' can select the static exception calling
  // convention, so the library support must be available.
  if (!CheckStaticExceptionSupport(Loc)) {
    EST = EST_None;
    return Cond;
  }

  if (Cond->isValueDependent() || Cond->containsUnexpandedParameterPack()) {
    EST = EST_DependentThrows;
    return Cond;
  }

  // The condition is an except_t: 0 (no_except), 1 (static_except) or 2
  // (dynamic_except); bool and other integral values are accepted too.
  if (!Cond->getType()->isIntegralOrEnumerationType()) {
    Diag(Cond->getBeginLoc(), diag::err_static_exception_condition_type)
        << Cond->getType() << Cond->getSourceRange();
    return ExprError();
  }
  std::optional<llvm::APSInt> Value = Cond->getIntegerConstantExpr(Context);
  if (!Value) {
    // Produce the usual diagnostics for a non-constant expression.
    (void)VerifyIntegerConstantExpression(Cond, /*Result=*/nullptr,
                                          AllowFoldKind::No);
    return ExprError();
  }
  if (*Value == 0) {
    EST = EST_BasicNoexcept;
  } else if (*Value == 1) {
    EST = EST_Throws;
  } else if (*Value == 2) {
    EST = EST_None;
  } else {
    Diag(Cond->getBeginLoc(), diag::err_static_exception_condition_value)
        << toString(*Value, 10) << Cond->getSourceRange();
    return ExprError();
  }
  return Cond;
}

ExprResult Sema::ActOnCXXExceptModeExpr(SourceLocation KeyLoc, SourceLocation,
                                        Expr *Operand, SourceLocation RParen) {
  return BuildCXXExceptModeExpr(KeyLoc, Operand, RParen);
}

QualType Sema::getStdExceptTType(SourceLocation Loc) {
  if (NamespaceDecl *Std = getStdNamespace()) {
    LookupResult R(*this, &PP.getIdentifierTable().get("except_t"), Loc,
                   LookupTagName);
    LookupQualifiedName(R, Std);
    if (auto *ED = R.getAsSingle<EnumDecl>(); ED && ED->isComplete())
      return Context.getCanonicalTagType(ED);
  }
  return Context.IntTy;
}

ExprResult Sema::BuildCXXExceptModeExpr(SourceLocation KeyLoc, Expr *Operand,
                                        SourceLocation RParen) {
  // The result has type std::except_t if the library declares it.
  QualType ResultTy = getStdExceptTType(KeyLoc);

  // throws(expr) is dynamic_except if expr can throw a dynamic exception,
  // else static_except if it can throw a static exception, else no_except.
  CanThrowResult CT = canThrow(Operand);
  bool Dependent = CT == CT_Dependent;
  unsigned Mode = CXXExceptModeExpr::NoExcept;
  if (CT == CT_Can) {
    llvm::SaveAndRestore IgnoreStatic(CanThrowIgnoresStaticExceptions, true);
    CanThrowResult DynamicCT = canThrow(Operand);
    Dependent = DynamicCT == CT_Dependent;
    Mode = DynamicCT == CT_Cannot ? CXXExceptModeExpr::StaticExcept
                                  : CXXExceptModeExpr::DynamicExcept;
  }
  return new (Context)
      CXXExceptModeExpr(ResultTy, Operand, Mode, Dependent, KeyLoc, RParen);
}

Decl *Sema::ActOnImplicitStaticCatchParameter(Scope *S, SourceLocation Loc) {
  if (!CheckStaticExceptionSupport(Loc))
    return nullptr;
  TypeSourceInfo *TInfo =
      Context.getTrivialTypeSourceInfo(Context.getStdErrorType(), Loc);
  VarDecl *ExDecl = BuildExceptionDeclaration(
      S, TInfo, Loc, Loc, &PP.getIdentifierTable().get("err"));
  ExDecl->setImplicit();
  PushOnScopeChains(ExDecl, S);
  return ExDecl;
}

void Sema::ActOnStandaloneCatch(Scope *S, SourceLocation CatchLoc) {
  // Move the local declarations (but not the parameters of a function body)
  // to a temporary scope and pop that, which also diagnoses unused ones.
  Scope TryScope(S, S->getFlags(), Diags);
  SmallVector<Decl *, 8> Decls(S->decls());
  for (Decl *D : Decls) {
    if (isa<ParmVarDecl>(D))
      continue;
    S->RemoveDecl(D);
    TryScope.AddDecl(D);
  }
  ActOnPopScope(CatchLoc, &TryScope);
}
