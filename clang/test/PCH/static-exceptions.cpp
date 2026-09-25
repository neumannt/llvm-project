// Test this without pch.
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fstatic-exceptions -fcxx-exceptions -fexceptions -include %s -emit-llvm -o - %s | FileCheck %s

// Test with pch.
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fstatic-exceptions -fcxx-exceptions -fexceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fstatic-exceptions -fcxx-exceptions -fexceptions -include-pch %t -emit-llvm -o - %s | FileCheck %s

// The same with the propagation hook.
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fstatic-exceptions -fstatic-exceptions-propagation-hook -fcxx-exceptions -fexceptions -emit-pch -o %t.hook %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fstatic-exceptions -fstatic-exceptions-propagation-hook -fcxx-exceptions -fexceptions -include-pch %t.hook -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,HOOK

// P0709: the std::error support that Sema sets up must be restored from the
// PCH even if nothing in this translation unit references a 'throws' function.

#ifndef HEADER
#define HEADER

namespace std {
struct error { const void *domain; long value; };
error __error_from_current_exception() noexcept;
[[noreturn]] void __throw_error_as_dynamic(error);
void __notify_error_propagation(const error &) noexcept;
}
extern int dom;
int mayThrow(int);

// Not referenced below, but emitted in every translation unit.
// CHECK-LABEL: define {{.*}} @_Z2pfi(
// HOOK: call void @_ZSt26__notify_error_propagationRKSt5error(
// CHECK: call {{.*}} @_ZSt30__error_from_current_exceptionv()
int pf(int x) throws {
  if (x)
    throw std::error{&dom, x};
  return mayThrow(x);
}

#else

int main() { return 0; }

#endif
