// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fstatic-exceptions -fstatic-exceptions-abi=pointer -fcxx-exceptions -fexceptions -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// A std::error that is not trivially copyable, but trivially relocatable
// ([[clang::trivial_abi]]), e.g. because it owns a wrapped dynamic exception.
// The compiler moves errors with memcpy, and copies and destroys them where
// the language requires it.

namespace std {
struct [[clang::trivial_abi]] error {
  const void *domain;
  long value;
  error(const error &) noexcept;
  error(error &&) noexcept;
  ~error();
};
error __error_from_current_exception() noexcept;
[[noreturn]] void __throw_error_as_dynamic(error);
} // namespace std

int leaf() throws;
void may_throw();
struct G { ~G(); };

// The catch parameter takes over the error from the slot and destroys it.
// CHECK-LABEL: define dso_local i32 @_Z8by_valuev(
// CHECK: static.catch:
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}%e, ptr {{.*}}%static.catch.slot, i64 16, i1 false)
// CHECK-NOT: _ZNSt5errorC1ERKS_
// CHECK: call void @_ZNSt5errorD1Ev(ptr {{.*}}%e)
int by_value() throws {
  try {
    return leaf();
  } catch (std::error e) {
    return 0;
  }
}

// 'throw;' runs the destructors in the handler first, then moves the catch
// parameter to the target instead of destroying it.
// CHECK-LABEL: define dso_local i32 @_Z7rethrowv(
// CHECK: static.catch:
// CHECK: call void @_ZN1GD1Ev(ptr {{.*}}%g)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}%static.error, ptr {{.*}}%e, i64 16, i1 false)
// CHECK-NEXT: br label %return
// CHECK-NOT: _ZNSt5errorC1ERKS_
// CHECK: ret i32
int rethrow() throws {
  try {
    return leaf();
  } catch (std::error e) {
    G g;
    throw;
  }
}

// A handler entered by a dynamic std::error copies it before ending the
// catch; a reference refers to the slot, which the handler destroys.
// CHECK-LABEL: define dso_local i32 @_Z9dyn_entryv(
// CHECK: {{^}}catch:
// CHECK: %[[OBJ:.*]] = call ptr @__cxa_begin_catch(
// CHECK-NEXT: call void @_ZNSt5errorC1ERKS_(ptr {{.*}}%static.catch.slot, ptr {{.*}}%[[OBJ]])
// CHECK-NEXT: call void @__cxa_end_catch()
// CHECK: call void @_ZNSt5errorD1Ev(ptr {{.*}}%static.catch.slot)
int dyn_entry() throws {
  try {
    may_throw();
    return leaf();
  } catch (const std::error &e) {
    return 0;
  }
}

// catch(...) destroys the slot if it was entered by a static exception.
// CHECK-LABEL: define dso_local i32 @_Z9catch_allv(
// CHECK: catch.end.dynamic:
// CHECK-NEXT: invoke void @__cxa_end_catch()
// CHECK: catch.end.static:
// CHECK-NEXT: call void @_ZNSt5errorD1Ev(ptr {{.*}}%static.catch.slot)
int catch_all() throws {
  try {
    may_throw();
    return leaf();
  } catch (...) {
    return 0;
  }
}

// Rethrowing as a dynamic exception copies the error: the catch parameter is
// destroyed during unwinding.
// CHECK-LABEL: define dso_local void @_Z10to_dynamicv(
// CHECK: static.catch:
// CHECK: call void @_ZNSt5errorC1ERKS_(ptr {{.*}}%static.error.tmp, ptr {{.*}}%e)
// CHECK: invoke void @_ZSt24__throw_error_as_dynamicSt5error(
// CHECK: call void @_ZNSt5errorD1Ev(ptr {{.*}}%e)
void to_dynamic() {
  try {
    leaf();
  } catch (std::error e) {
    throw;
  }
}

// 'throw;' to a try statement inside the handler: the catch parameter outlives
// the exit, so the error is moved out of it (leaving a valid error) instead of
// being relocated, and destroyed only once, at the end of the handler.
// CHECK-LABEL: define dso_local i32 @_Z14nested_rethrowv(
// CHECK: static.catch:
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}%e, ptr {{.*}}%static.catch.slot, i64 16, i1 false)
// CHECK-NOT: _ZNSt5errorC1ERKS_
// CHECK: call void @_ZNSt5errorC1EOS_(ptr {{.*}}%static.catch.slot{{[0-9]+}}, ptr {{.*}}%e)
// CHECK-NOT: _ZNSt5errorD1Ev(ptr {{.*}}%e)
// CHECK: call void @_ZNSt5errorD1Ev(ptr {{.*}}%e2)
// CHECK-NOT: _ZNSt5errorD1Ev(ptr {{.*}}%e)
// CHECK: call void @_ZNSt5errorD1Ev(ptr {{.*}}%e)
// CHECK-NOT: _ZNSt5errorD1Ev(ptr {{.*}}%e)
// CHECK: ret i32
int nested_rethrow() throws {
  try {
    leaf();
  } catch (std::error e) {
    try {
      throw;
    } catch (std::error e2) {
    }
  }
  return 0;
}
