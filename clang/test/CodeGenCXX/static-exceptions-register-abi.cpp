// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fstatic-exceptions -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,X64,REG
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fstatic-exceptions -fstatic-exceptions-abi=register -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,X64,REG
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fstatic-exceptions -fstatic-exceptions-abi=carry -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,X64,CARRY
// RUN: %clang_cc1 -triple aarch64-linux-gnu -std=c++17 -fstatic-exceptions -fstatic-exceptions-abi=register -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,A64,REG
// RUN: %clang_cc1 -triple i686-linux-gnu -std=c++17 -fstatic-exceptions -fstatic-exceptions-abi=carry -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=I686

// P0709 static exceptions with the error returned in registers
// (-fstatic-exceptions-abi=register/carry): a 'throws' function returns a
// struct of words that hold the value or the error, other parts of the value,
// and a trailing i1 that is true on failure.

namespace std {
struct error { const void *domain; long value; };
error __error_from_current_exception() noexcept;
[[noreturn]] void __throw_error_as_dynamic(error);
}
extern int dom;
struct Big { long a, b, c, d; };

// No hidden parameter. A static throw returns the error in the first two
// words and sets the flag.
// CHECK-LABEL: define dso_local { i64, i64, i1 } @_Z4leafi(i32 noundef %x)
// CHECK-SAME: #[[THROWS:[0-9]+]]
// CHECK: static.error.return:
// CHECK: %[[DOM:.*]] = load ptr, ptr %static.error
// CHECK: %[[VAL:.*]] = load i64, ptr
// CHECK: %[[DOMW:.*]] = ptrtoint ptr %[[DOM]] to i64
// CHECK: %[[R0:.*]] = insertvalue { i64, i64, i1 } poison, i64 %[[DOMW]], 0
// CHECK: %[[R1:.*]] = insertvalue { i64, i64, i1 } %[[R0]], i64 %[[VAL]], 1
// CHECK: %[[R2:.*]] = insertvalue { i64, i64, i1 } %[[R1]], i1 true, 2
// CHECK: ret { i64, i64, i1 } %[[R2]]
// A successful return puts the value into the first word.
// CHECK: ret { i64, i64, i1 } { i64 1, i64 poison, i1 false }
// I686-LABEL: define dso_local { i32, i32, i1 } @_Z4leafi(
int leaf(int x) throws {
  if (x)
    throw std::error{&dom, x};
  return 1;
}

// After a call, the caller tests the flag, and on failure stores the error to
// the slot of its target.
// CHECK-LABEL: define dso_local { i64, i64, i1 } @_Z9propagatei(
// CHECK: %call = call { i64, i64, i1 } @_Z4leafi(i32 noundef %{{.*}})
// CHECK-NEXT: %static.failed = extractvalue { i64, i64, i1 } %call, 2
// CHECK-NEXT: br i1 %static.failed, label %static.unwind, label %static.cont, !prof ![[UNLIKELY:[0-9]+]]
// CHECK: static.unwind:
// CHECK-NEXT: %[[W0:.*]] = extractvalue { i64, i64, i1 } %call, 0
// CHECK-NEXT: %[[D:.*]] = inttoptr i64 %[[W0]] to ptr
// CHECK-NEXT: store ptr %[[D]], ptr %static.error
// CHECK-NEXT: %[[W1:.*]] = extractvalue { i64, i64, i1 } %call, 1
// CHECK-NEXT: %[[P:.*]] = getelementptr inbounds i8, ptr %static.error, i64 8
// CHECK-NEXT: store i64 %[[W1]], ptr %[[P]]
// CHECK-NEXT: br label %static.error.return
// CHECK: static.cont:
// CHECK-NEXT: %[[V:.*]] = extractvalue { i64, i64, i1 } %call, 0
// CHECK-NEXT: trunc i64 %[[V]] to i32
int propagate(int x) throws { return leaf(x) + 1; }

// 'return f(...)' returns the result as is (and becomes a tail call).
// CHECK-LABEL: define dso_local { i64, i64, i1 } @_Z7forwardi(
// CHECK: %call = call { i64, i64, i1 } @_Z4leafi(
// CHECK-NEXT: ret { i64, i64, i1 } %call
int forward(int x) throws { return leaf(x); }

// Not if cleanups have to run.
struct S { S(); ~S(); };
// CHECK-LABEL: define dso_local { i64, i64, i1 } @_Z15forward_cleanupi(
// CHECK: %call = call { i64, i64, i1 } @_Z4leafi(
// CHECK-NEXT: %static.failed = extractvalue
int forward_cleanup(int x) throws { S s; return leaf(x); }

// CHECK-LABEL: define dso_local { i64, i64, i1 } @_Z7nothingv()
// CHECK: ret { i64, i64, i1 } { i64 poison, i64 poison, i1 false }
void nothing() throws {}

// A floating-point value keeps its register; the error uses the words.
// X64: declare { i64, i64, double, i1 } @_Z3dblv()
// A64: declare { i64, i64, double, i1 } @_Z3dblv()
// I686: declare { i32, i32, double, i1 } @_Z3dblv()
double dbl() throws;
double usedbl() throws { return dbl() * 2; }

// Pointers keep their type.
// CHECK: declare { ptr, i64, i1 } @_Z3ptrv()
// I686: declare { ptr, i32, i1 } @_Z3ptrv()
int *ptr() throws;
int useptr() throws { return *ptr(); }

// Pointers in other address spaces do not share the words with the error.
// CHECK-LABEL: define dso_local { i64, i64, ptr addrspace(256), i1 } @_Z5asptri(
// CHECK: ret { i64, i64, ptr addrspace(256), i1 }
// CHECK-LABEL: define dso_local { i64, i64, i1 } @_Z8useasptrv(
// CHECK: call { i64, i64, ptr addrspace(256), i1 } @_Z5asptri(
// I686-LABEL: define dso_local { i32, i32, ptr addrspace(256), i1 } @_Z5asptri(
int __attribute__((address_space(256))) *asptr(int x) throws {
  if (x)
    throw std::error{&dom, x};
  return nullptr;
}
int useasptr() throws { return *asptr(1); }

// Integers that are wider than a word take several words.
// X64: declare { i64, i64, i1 } @_Z2llv()
// I686: declare { i32, i32, i1 } @_Z2llv()
long long ll() throws;
long long usell() throws { return ll() + 1; }

// A value returned in memory still is, but the pointer is not 'sret'.
// X64-LABEL: define dso_local { i64, i64, i1 } @_Z6usebigv(
// X64: call { i64, i64, i1 } @_Z3bigv(ptr dead_on_unwind noalias writable align 8 %{{.*}})
// X64: declare { i64, i64, i1 } @_Z3bigv(ptr dead_on_unwind noalias writable align 8)
Big big() throws;
long usebig() throws { return big().c; }

// Forwarding a value in memory passes our own return slot.
// X64-LABEL: define dso_local { i64, i64, i1 } @_Z6fwdbigv(ptr dead_on_unwind noalias writable align 8 %agg.result)
// X64: %call = call { i64, i64, i1 } @_Z3bigv(ptr dead_on_unwind noalias writable align 8 %agg.result)
// X64-NEXT: ret { i64, i64, i1 } %call
Big fwdbig() throws { return big(); }

// A local handler receives the error from the flag check.
// CHECK-LABEL: define dso_local {{.*}}i32 @_Z7handleri(
// CHECK: %call = call { i64, i64, i1 } @_Z4leafi(
// CHECK: br i1 %static.failed, label %static.unwind, label %static.cont
// CHECK: static.unwind:
// CHECK: store ptr %{{.*}}, ptr %static.catch.slot
// CHECK: br label %static.catch
int handler(int x) noexcept {
  try {
    return leaf(x);
  } catch (std::error e) {
    return (int)e.value;
  }
}

// CARRY: attributes #[[THROWS]] = {{{.*}}nounwind{{.*}}"x86-carry-flag-return"
// REG: attributes #[[THROWS]] = {
// REG-NOT: x86-carry-flag-return
// CHECK: ![[UNLIKELY]] = !{!"branch_weights", {{.*}}i32 1, i32 1048575}
