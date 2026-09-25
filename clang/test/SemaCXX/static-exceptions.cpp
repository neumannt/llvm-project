// RUN: %clang_cc1 -std=c++17 -fstatic-exceptions -fcxx-exceptions -fexceptions -triple x86_64-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++17 -fstatic-exceptions -triple x86_64-linux-gnu -fsyntax-only -verify=expected,noexc %s

// Tests for the P0709 static exceptions prototype ('throws').

int before_error() throws; // expected-error {{use of 'throws' requires a definition of 'std::error'; include <error>}}

namespace std {
struct error {
  const void *domain;
  long value;
};
struct error_from_errc {
  operator error() const noexcept { return {this, 1}; }
};
error __error_from_current_exception() noexcept;
[[noreturn]] void __throw_error_as_dynamic(error);
} // namespace std

// Declarations and types.
int f() throws;
int f() throws;
int f(); // expected-error {{exception specification in declaration does not match previous declaration}}
         // expected-note@-2 {{previous declaration is here}}

void g();
void g() throws; // expected-error {{exception specification in declaration does not match previous declaration}}
                 // expected-note@-2 {{previous declaration is here}}

int h() noexcept throws; // expected-error {{'throws' cannot be combined with another exception specification}}
int h2() throws noexcept; // expected-error {{'throws' cannot be combined with another exception specification}}

// 'throws' behaves as noexcept(false) for the noexcept operator.
static_assert(!noexcept(f()));

// 'throws' is part of the function type.
using F = int() throws;
static_assert(!__is_same(F, int()));
F *fp1 = f;
int (*fp2)() = f; // expected-error {{cannot initialize a variable of type 'int (*)()' with an lvalue of type 'int () throws'}}
int (*fp3)() throws = f;
int nothrow_fn() noexcept;
int (*fp4)() throws = nothrow_fn; // expected-error {{cannot initialize a variable of type 'int (*)() throws' with an lvalue of type 'int () noexcept'}}

// Virtual functions must agree.
struct B {
  virtual int v1() throws;
  virtual int v2();
  virtual int v3() throws;
};
struct D : B {
  int v1() throws override;
  int v2() throws override; // expected-error {{function declared 'throws' cannot override a function that is not declared 'throws'}}
                            // expected-note@-5 {{overridden virtual function is here}}
  int v3() override;        // expected-error {{function overriding a 'throws' function must also be declared 'throws'}}
                            // expected-note@-8 {{overridden virtual function is here}}
};

struct Dtor {
  ~Dtor() throws; // expected-error {{destructor cannot be declared 'throws'}}
};

// Lambdas.
auto lam = []() throws { return 1; };
static_assert(!noexcept(lam()));
int (*lamp)() throws = lam;

// Templates.
template <class T> T tf(T t) throws { if (!t) throw std::error_from_errc{}; return t; }
template <class T> struct TS { T m() throws; };
int use_templates() throws { return tf(1) + TS<int>().m(); }

// Throwing values.
int thrower(int x) throws {
  if (x == 0)
    throw std::error{nullptr, 1};
  if (x == 1)
    throw std::error_from_errc{};
  if (x == 2)
    throw 42; // noexc-error {{cannot use 'throw' with exceptions disabled}}
  try {
    thrower(x - 1);
  } catch (std::error e) {
    throw;
  } catch (...) {
    throw;
  }
  return x;
}

void not_throws() {
  throw std::error{}; // noexc-error {{cannot use 'throw' with exceptions disabled}}
}

// Outside of a handler, 'throw;' rethrows a dynamic exception.
void rethrow_outside_handler() {
  throw; // noexc-error {{cannot use 'throw' with exceptions disabled}}
}
void rethrow_in_lambda_in_handler() {
  try {
    f();
  } catch (...) {
    [] { throw; }(); // noexc-error {{cannot use 'throw' with exceptions disabled}}
  }
}

// 'throws' functions have their own calling convention.
int mt_dynamic();
int mt_static() throws;
int mt_from_static() throws {
  [[clang::musttail]] return mt_dynamic(); // expected-error {{cannot perform a tail call from a function declared 'throws' to a function that is not}} expected-note {{tail call required by 'clang::musttail' attribute here}}
}
int mt_to_static() {
  [[clang::musttail]] return mt_static(); // expected-error {{cannot perform a tail call to a function declared 'throws' from a function that is not}} expected-note {{tail call required by 'clang::musttail' attribute here}}
}
int mt_both_static() throws { [[clang::musttail]] return mt_static(); }

// Conditional static exception specifications (P0709 4.1.4).
namespace std {
enum except_t { no_except = false, static_except = true, dynamic_except };
}
int n0() noexcept;
int d0();
static_assert(throws(n0()) == std::no_except);
static_assert(throws(f()) == std::static_except);
static_assert(throws(d0()) == std::dynamic_except);
static_assert(throws(f() + d0()) == std::dynamic_except);
static_assert(__is_same(decltype(throws(f())), std::except_t));

int c0() throws(std::no_except);
int c1() throws(std::static_except);
int c2() throws(std::dynamic_except);
static_assert(__is_same(decltype(c0), int() noexcept));
static_assert(__is_same(decltype(c1), int() throws));
static_assert(__is_same(decltype(c2), int()));
int c3() throws(3); // expected-error {{condition of 'throws' has value 3; expected 0 (no_except), 1 (static_except) or 2 (dynamic_except)}}
int c4() throws(1.0); // expected-error {{condition of 'throws' must have integral or enumeration type, not 'double'}}
int c1() throws; // OK, same as throws(std::static_except)

template <class F> int apply(F fn) throws(throws(fn())) { return fn(); }
int apply_static() throws { return apply(f); }
static_assert(noexcept(apply(n0)));
static_assert(throws(apply(f)) == std::static_except);
static_assert(throws(apply(d0)) == std::dynamic_except);

// With a dependent 'throws(cond)', whether a throw-expression throws a static
// exception is only known after instantiation.
template <std::except_t E> int dep_throw() throws(E) {
  throw std::error_from_errc{}; // noexc-error {{cannot use 'throw' with exceptions disabled}}
}
int use_dep_throw() throws { return dep_throw<std::static_except>(); }
int use_dep_throw_dynamic() {
  return dep_throw<std::dynamic_except>(); // noexc-note {{in instantiation of function template specialization 'dep_throw<std::dynamic_except>' requested here}}
}

template <class T> struct CT {
  virtual void v() throws(T::value); // expected-error {{virtual function cannot have a conditional 'throws' specification}}
  void m() throws(T::value);
};

// 4.5: try-expressions, 'catch {}' and standalone 'catch'.
int try_exprs(int x) throws {
  int a = try f();
  try int b = f();
  try return a + b + try f() + x;
}

int sugar() {
  int r = try f();
  return r;
  catch {
    return err.value;
  }
}

int standalone_scope() {
  int local = f();
  return local;
  catch (...) {
    return local; // expected-error {{use of undeclared identifier 'local'}}
  }
}

int standalone_last() {
  f();
  catch {
  }
  return 1; // expected-error {{a standalone 'catch' must be at the end of its block}}
}
