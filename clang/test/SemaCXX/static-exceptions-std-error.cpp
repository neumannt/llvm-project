// RUN: %clang_cc1 -std=c++17 -fstatic-exceptions -fcxx-exceptions -fexceptions -triple x86_64-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++17 -fstatic-exceptions -fcxx-exceptions -fexceptions -triple x86_64-linux-gnu -fsyntax-only -verify=ok -DTRIVIAL_ABI %s

// std::error must be trivially relocatable: trivially copyable, or
// [[clang::trivial_abi]].

// ok-no-diagnostics

namespace std {
#ifdef TRIVIAL_ABI
struct [[clang::trivial_abi]] error {
#else
struct error { // expected-error {{'std::error' must be a trivially copyable or [[clang::trivial_abi]], non-polymorphic class without base classes whose first member is a pointer}}
#endif
  const void *domain;
  long value;
  error(const error &) noexcept;
  ~error();
};
error __error_from_current_exception() noexcept;
[[noreturn]] void __throw_error_as_dynamic(error);
} // namespace std

int f() throws;
