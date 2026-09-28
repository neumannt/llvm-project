// RUN: %clang_cc1 -std=c++20 -triple x86_64-linux-gnu -fstatic-exceptions -fstatic-exceptions-abi=pointer -fcxx-exceptions -fexceptions -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// P0709: the implicit handler of a coroutine calls unhandled_exception(),
// which expects a current exception. Static exceptions raised in the body
// are thrown as dynamic exceptions, so that the handler can catch them.

#include "Inputs/coroutine.h"

namespace std {
struct error { const void *domain; long value; };
error __error_from_current_exception() noexcept;
[[noreturn]] void __throw_error_as_dynamic(error);
} // namespace std

struct task {
  struct promise_type {
    task get_return_object();
    std::suspend_never initial_suspend() noexcept;
    std::suspend_never final_suspend() noexcept;
    void return_void();
    void unhandled_exception();
  };
};

int fail() throws;

// CHECK-LABEL: define dso_local void @_Z2cov(
// CHECK: call i32 @_Z4failv(ptr {{.*}} %static.error.tmp)
// CHECK: static.unwind:
// CHECK: invoke void @_ZSt24__throw_error_as_dynamicSt5error(
// CHECK-NEXT: to label %{{.*}} unwind label %[[LPAD:.*]]
// CHECK: [[LPAD]]:
// CHECK: invoke void @_ZN4task12promise_type19unhandled_exceptionEv(
task co() {
  fail();
  co_return;
}
