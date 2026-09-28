// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fstatic-exceptions -fstatic-exceptions-abi=pointer -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,PTR
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fstatic-exceptions -fstatic-exceptions-abi=register -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,REG

// Function attributes that conflict with how a P0709 'throws' function
// reports failure.

namespace std {
struct error { const void *domain; long value; };
error __error_from_current_exception() noexcept;
[[noreturn]] void __throw_error_as_dynamic(error);
}
extern int dom;
void f(int) throws;
[[noreturn]] void abort_it();

// With the pointer ABI, a const or pure 'throws' function writes its failure
// through the std::error out-parameter, so it must not be readnone or
// readonly. With the register ABI the error is part of the return value.
// PTR: define dso_local i32 @_Z2sqi(i32 noundef %x, ptr noalias noundef nonnull align 8 dereferenceable(16) %static.error) #[[CONST:[0-9]+]]
// REG: define dso_local { i64, i64, i1 } @_Z2sqi(i32 noundef %x) #[[CONST:[0-9]+]]
[[gnu::const]] int sq(int x) throws {
  if (x < 0)
    throw std::error{&dom, x};
  return x * x;
}

// PTR: define dso_local i32 @_Z2rdPKi(ptr noundef readonly %p, ptr noalias noundef nonnull align 8 dereferenceable(16) %static.error) #[[PURE:[0-9]+]]
// REG: define dso_local { i64, i64, i1 } @_Z2rdPKi(ptr noundef %p) #[[PURE:[0-9]+]]
[[gnu::pure]] int rd(const int *p) throws {
  if (!*p)
    throw std::error{&dom, 1};
  return *p;
}

// A GNU noreturn attribute is part of the function type. The function still
// returns when it fails, so its return block must not be unreachable.
// CHECK-LABEL: define dso_local {{.*}}@_Z2nri(
// CHECK: static.unwind:
// CHECK-NOT: unreachable
// PTR: ret void
// REG: ret { i64, i64, i1 }
__attribute__((noreturn)) void nr(int v) throws {
  f(v);
  abort_it();
}

// PTR: attributes #[[CONST]] = { {{.*}}memory(argmem: readwrite){{.*}} }
// PTR: attributes #[[PURE]] = { {{.*}}memory(read, argmem: readwrite){{.*}} }
// REG: attributes #[[CONST]] = { {{.*}}memory(none){{.*}} }
// REG: attributes #[[PURE]] = { {{.*}}memory(read){{.*}} }
