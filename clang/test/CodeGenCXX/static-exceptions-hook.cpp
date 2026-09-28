// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fstatic-exceptions -fstatic-exceptions-abi=pointer -fstatic-exceptions-propagation-hook -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,HOOK
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fstatic-exceptions -fstatic-exceptions-abi=pointer -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,NOHOOK

namespace std {
struct error { const void *domain; long value; };
[[noreturn]] void __throw_error_as_dynamic(error);
void __notify_error_propagation(const error &) noexcept;
enum except_t { no_except, static_except, dynamic_except };
}
extern int dom;
int leaf(int x) throws;

// All error exits share one call of the propagation hook (P0709 4.4).
// CHECK-LABEL: define dso_local i32 @_Z3midi(
// CHECK: static.unwind:
// HOOK-NEXT: br label %static.error.return
// NOHOOK-NEXT: br label %return
// HOOK: static.error.return:
// HOOK-NEXT: call void @_ZSt26__notify_error_propagationRKSt5error(ptr {{.*}} %static.error)
// HOOK-NEXT: br label %return
// NOHOOK-NOT: __notify_error_propagation
int mid(int x) throws {
  if (x > 10)
    throw std::error{&dom, 1};
  return leaf(x) + leaf(x + 1);
}

// A conditional 'throws' specification selects the calling convention on
// instantiation.
// CHECK-LABEL: define linkonce_odr i32 @_Z5applyIPU6throwsFiiEEiT_(ptr noundef %f, ptr noalias
template <class F> int apply(F f) throws(throws(f(1))) { return f(1); }
int use() throws { return apply(leaf); }

// A dependent conditional 'throws' in a function type is mangled as a vendor
// qualifier with the condition as template argument.
// CHECK-LABEL: define linkonce_odr void @_Z5takesI3TagEvPU6throwsIXsrT_5valueEEFiiE(
struct Tag { static constexpr std::except_t value = std::static_except; };
template <class T> void takes(int (*)(int) throws(T::value)) {}
void use_takes() { takes<Tag>(leaf); }

// A this-adjusting thunk forwards the error of the function it calls, which
// already called the hook.
// CHECK-LABEL: define dso_local void @_ZThn16_N1C1gEv(
// CHECK-NOT: __notify_error_propagation
// CHECK: ret void
struct TA { virtual ~TA(); int x; };
struct TB { virtual void g() throws = 0; };
struct C : TA, TB { void g() throws override; };
void C::g() throws { leaf(0); }
