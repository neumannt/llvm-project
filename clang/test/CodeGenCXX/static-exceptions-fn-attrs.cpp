// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fstatic-exceptions -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

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

// A const or pure 'throws' function writes its failure through the std::error
// out-parameter, so it must not be readnone or readonly.
// CHECK: define dso_local i32 @_Z2sqi(i32 noundef %x, ptr noalias noundef nonnull align 8 dereferenceable(16) %static.error) #[[CONST:[0-9]+]]
[[gnu::const]] int sq(int x) throws {
  if (x < 0)
    throw std::error{&dom, x};
  return x * x;
}

// CHECK: define dso_local i32 @_Z2rdPKi(ptr noundef readonly %p, ptr noalias noundef nonnull align 8 dereferenceable(16) %static.error) #[[PURE:[0-9]+]]
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
// CHECK: ret void
__attribute__((noreturn)) void nr(int v) throws {
  f(v);
  abort_it();
}

// CHECK: attributes #[[CONST]] = { {{.*}}memory(argmem: readwrite){{.*}} }
// CHECK: attributes #[[PURE]] = { {{.*}}memory(read, argmem: readwrite){{.*}} }
