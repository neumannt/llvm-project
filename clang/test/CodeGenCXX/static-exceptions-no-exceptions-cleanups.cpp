// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fstatic-exceptions -fstatic-exceptions-abi=pointer -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// Without -fexceptions, CodeGen normally pushes no EH-only cleanups. A static
// exception still needs them: its error path runs them inline.

namespace std {
struct error { const void *domain; long value; };
[[noreturn]] void __throw_error_as_dynamic(error);
} // namespace std

int fail() throws;
struct Base { Base(); ~Base(); };
struct A { A(); ~A(); };
struct S : Base {
  A a;
  int b;
  S() throws : a(), b(fail()) {}
};

// Delegating constructor: the object is destroyed if the body fails.
struct Del {
  A a;
  Del() throws : Del(0) { fail(); }
  Del(int);
};

// Array elements constructed before the failure are destroyed.
// CHECK-LABEL: define dso_local void @_Z5arrayv(
// CHECK: call void @_ZN1TC1Ev(
// CHECK: static.unwind:
// CHECK: arraydestroy.body
// CHECK: call void @_ZN1TD1Ev(
struct T { T() throws; ~T(); };
void array() throws { T t[4]; }

// Aggregate members initialized before the failure are destroyed.
// CHECK-LABEL: define dso_local void @_Z3aggv(
// CHECK: call void @_ZN1AC1Ev(
// CHECK: call void @_ZN1AC1Ev(
// CHECK: call i32 @_Z4failv(
// CHECK: static.unwind:
// CHECK: call void @_ZN1AD1Ev(
// CHECK: call void @_ZN1AD1Ev(
struct Agg { A x, y; int z; };
void agg() throws { Agg a{A(), A(), fail()}; }

void use() throws {
  S s;
  Del d;
}

// Emitted after use():
// CHECK-LABEL: define linkonce_odr void @_ZN3DelC1Ev(
// CHECK: call void @_ZN3DelC1Ei(
// CHECK: call i32 @_Z4failv(
// CHECK: static.unwind:
// CHECK: call void @_ZN3DelD1Ev(

// Members and bases constructed before the failure are destroyed.
// CHECK-LABEL: define linkonce_odr void @_ZN1SC2Ev(
// CHECK: call void @_ZN4BaseC2Ev(
// CHECK: call void @_ZN1AC1Ev(
// CHECK: call i32 @_Z4failv(
// CHECK: static.unwind:
// CHECK: call void @_ZN1AD1Ev(
// CHECK: call void @_ZN4BaseD2Ev(
