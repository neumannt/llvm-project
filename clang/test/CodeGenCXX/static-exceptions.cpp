// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fstatic-exceptions -fcxx-exceptions -fexceptions -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,EH
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fstatic-exceptions -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,NOEH

// Code generation for the P0709 static exceptions prototype ('throws').

namespace std {
struct error { const void *domain; long value; };
error __error_from_current_exception() noexcept;
[[noreturn]] void __throw_error_as_dynamic(error);
}
extern int dom;
struct S { S(); ~S(); };

// A 'throws' function takes a hidden std::error out-parameter, is nounwind,
// and has no value attributes on its return value.
// CHECK-LABEL: define dso_local i32 @_Z4leafi(i32 noundef %x, ptr noalias noundef nonnull align 8 dereferenceable(16) %static.error)
// CHECK-SAME: #[[THROWS_ATTRS:[0-9]+]]
// A static throw stores the error into the out-parameter and returns.
// CHECK: if.then:
// CHECK: %[[DOM:.*]] = getelementptr inbounds nuw %"struct.std::error", ptr %static.error, i32 0, i32 0
// CHECK: store ptr @dom, ptr %[[DOM]]
// CHECK: br label %return
int leaf(int x) throws {
  if (x)
    throw std::error{&dom, x};
  return 1;
}

// Propagation passes the caller's own out-parameter down and checks the
// domain pointer after the call.
// CHECK-LABEL: define dso_local i32 @_Z9propagatei(
// CHECK: %call = call i32 @_Z4leafi(i32 noundef %{{.*}}, ptr noalias noundef nonnull align 8 dereferenceable(16) %static.error)
// CHECK-NEXT: %[[D:.*]] = load ptr, ptr %static.error
// CHECK-NEXT: %[[F:.*]] = icmp ne ptr %[[D]], null
// CHECK-NEXT: br i1 %[[F]], label %static.unwind, label %static.cont, !prof ![[UNLIKELY:[0-9]+]]
// CHECK: static.unwind:
// CHECK-NEXT: br label %return
int propagate(int x) throws { return leaf(x) + 1; }

// A local catch(std::error) receives static exceptions by a direct branch.
// CHECK-LABEL: define dso_local noundef i32 @_Z6handlei(
// CHECK: store ptr null, ptr %static.catch.slot
// CHECK-NEXT: %call = call i32 @_Z4leafi(i32 noundef %{{.*}}, ptr {{.*}} %static.catch.slot)
// CHECK: br i1 %{{.*}}, label %static.unwind, label %static.cont
// CHECK: static.unwind:
// CHECK-NEXT: br label %static.catch
// CHECK: static.catch:
// CHECK-NEXT: br label %catch.body
// CHECK: catch.body:
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %e, ptr align 8 %static.catch.slot, i64 16, i1 false)
int handle(int x) {
  try {
    return leaf(x);
  } catch (std::error e) {
    return (int)e.value;
  }
}

// In a function without 'throws', an unhandled static exception becomes a
// dynamic exception.
// CHECK-LABEL: define dso_local noundef i32 @_Z10to_dynamici(
// CHECK: call i32 @_Z4leafi(i32 noundef %{{.*}}, ptr {{.*}} %static.error.tmp)
// CHECK: static.unwind:
// CHECK: call void @_ZSt24__throw_error_as_dynamicSt5error(
// CHECK-NEXT: unreachable
int to_dynamic(int x) { return leaf(x); }

// Cleanups run inline on the static exception path.
// CHECK-LABEL: define dso_local i32 @_Z12with_cleanupi(
// CHECK: call i32 @_Z4leafi(
// CHECK: static.unwind:
// CHECK-NEXT: call void @_ZN1SD1Ev(ptr {{.*}} %s)
// CHECK-NEXT: br label %return
// CHECK: static.cont:
// CHECK: call void @_ZN1SD1Ev(ptr {{.*}} %s)
int with_cleanup(int x) throws {
  S s;
  return leaf(x);
}

// Dynamic exceptions escaping a 'throws' function are translated.
void may_throw();
// CHECK-LABEL: define dso_local i32 @_Z9translatev(
// EH-SAME: personality ptr @__gxx_personality_v0
// EH: invoke void @_Z9may_throwv()
// EH: static.translate:
// EH: call ptr @__cxa_begin_catch(
// EH: call { ptr, i64 } @_ZSt30__error_from_current_exceptionv()
// EH: invoke void @__cxa_end_catch()
// NOEH: call void @_Z9may_throwv()
// NOEH-NOT: landingpad
// CHECK: ret i32
int translate() throws {
  may_throw();
  return 0;
}

// The error parameter follows the fixed parameters of a variadic function.
// CHECK-LABEL: define dso_local i32 @_Z13call_variadicv(
// CHECK: call i32 (i32, ptr, ...) @_Z8variadiciz(i32 noundef 1, ptr noalias noundef nonnull align 8 dereferenceable(16) %static.error, i32 noundef 2, i32 noundef 3)
int variadic(int n, ...) throws;
int call_variadic() throws { return variadic(1, 2, 3); }

// 'throws' is part of the mangled function type.
// CHECK-LABEL: define dso_local void @_Z8takes_fpPU6throwsFiiE(
void takes_fp(int (*)(int) throws) {}

// catch(...) also receives static exceptions; 'throw;' rethrows the kind of
// exception that was caught.
// CHECK-LABEL: define dso_local i32 @_Z9catch_alli(
// EH: store i1 true, ptr %catch.is.dynamic
// CHECK: static.catch:
// EH: store i1 false, ptr %catch.is.dynamic
// CHECK: catch.body:
// EH: rethrow.dynamic:
// EH: {{call|invoke}} void @__cxa_rethrow()
// EH: rethrow.static:
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %static.error, ptr align 8 %static.catch.slot, i64 16, i1 false)
int catch_all(int x) throws {
  try {
    may_throw();
    return leaf(x);
  } catch (...) {
    throw;
  }
}

// If only static exceptions can reach a catch(...), there is no dynamic
// entry and no landing pad.
// CHECK-LABEL: define dso_local i32 @_Z16catch_all_statici(
// CHECK-NOT: landingpad
// CHECK-NOT: __cxa_rethrow
// CHECK: ret i32
int catch_all_static(int x) throws {
  try {
    return leaf(x);
  } catch (...) {
    throw;
  }
}

// A static exception escaping a cleanup that runs on the static error path
// terminates, just like a dynamic exception escaping it.
// CHECK-LABEL: define dso_local i32 @_Z14cleanup_throwsi(
// CHECK: static.unwind:
// CHECK: call void @_Z10cleanup_fnPi(ptr noundef %y, ptr {{.*}} %static.error.tmp)
// CHECK: static.unwind{{[0-9]+}}:
// CHECK: call void @_ZSt9terminatev()
// CHECK-NEXT: unreachable
void cleanup_fn(int *) throws;
int cleanup_throws(int x) throws {
  int y __attribute__((cleanup(cleanup_fn))) = x;
  return leaf(x);
}

// A musttail call between 'throws' functions skips the translation of
// dynamic exceptions, which the callee does itself.
// CHECK-LABEL: define dso_local i32 @_Z13musttail_nexti(
// CHECK: musttail call i32 @_Z4leafi(i32 noundef %{{.*}}, ptr {{.*}} %static.error)
// CHECK-NEXT: ret i32
int musttail_next(int x) throws {
  may_throw();
  [[clang::musttail]] return leaf(x);
}

// With a dependent 'throws(cond)', a throw-expression is static if the
// instantiation is 'throws'.
int use_dep_throw(int x) throws;
template <bool B> int dep_throw(int x) throws(B) {
  if (x)
    throw std::error{&dom, x};
  return 0;
}
int use_dep_throw(int x) throws { return dep_throw<true>(x); }
// CHECK-LABEL: define linkonce_odr i32 @_Z9dep_throwILb1EEii(
// CHECK-NOT: __cxa_throw
// CHECK: store ptr @dom
// CHECK: ret i32

// CHECK: attributes #[[THROWS_ATTRS]] = { {{.*}}nounwind
// CHECK: ![[UNLIKELY]] = !{!"branch_weights", i32 1, i32 {{[0-9]+}}}
