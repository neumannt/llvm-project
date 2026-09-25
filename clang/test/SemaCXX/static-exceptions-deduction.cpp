// RUN: %clang_cc1 -std=c++17 -fstatic-exceptions -fcxx-exceptions -fexceptions -triple x86_64-linux-gnu -fsyntax-only -verify %s

// P0709: template argument deduction through a conditional static exception
// specification throws(M).

namespace std {
struct error {
  const void *domain;
  long value;
};
error __error_from_current_exception() noexcept;
[[noreturn]] void __throw_error_as_dynamic(error);
enum except_t { no_except = false, static_except = true, dynamic_except };
} // namespace std

void fn() noexcept;
void fs() throws;
void fd();

// Deduce the mode as std::except_t.
template <std::except_t M> constexpr std::except_t mode(void (*)() throws(M)) {
  return M;
}
static_assert(mode(fn) == std::no_except);
static_assert(mode(fs) == std::static_except);
static_assert(mode(fd) == std::dynamic_except);

// ... as int.
template <int M> constexpr int imode(void (*)() throws(M)) { return M; }
static_assert(imode(fn) == 0);
static_assert(imode(fs) == 1);
static_assert(imode(fd) == 2);

// ... as auto.
template <auto M> constexpr auto amode(void (*)() throws(M)) { return M; }
static_assert(amode(fs) == std::static_except);
static_assert(__is_same(decltype(amode(fs)), std::except_t));

// A bool cannot represent dynamic_except.
template <bool M> constexpr bool bmode(void (*)() throws(M)) { return M; } // expected-note {{candidate template ignored}}
static_assert(!bmode(fn));
static_assert(bmode(fs));
bool b = bmode(fd); // expected-error {{no matching function for call to 'bmode'}}

// throws(M) is not deduced like noexcept(M), and vice versa.
template <bool B> constexpr bool nmode(void (*)() noexcept(B)) { return B; } // expected-note {{candidate template ignored}}
static_assert(nmode(fn));
static_assert(!nmode(fd));
bool n = nmode(fs); // expected-error {{no matching function for call to 'nmode'}}

// Deduction from a function type of a template.
template <std::except_t M> struct Wrap {
  static void f() throws(M);
};
static_assert(mode(Wrap<std::no_except>::f) == std::no_except);
static_assert(mode(Wrap<std::static_except>::f) == std::static_except);
static_assert(mode(Wrap<std::dynamic_except>::f) == std::dynamic_except);

// Partial ordering deduces from a dependent throws(M).
template <std::except_t M, class T> constexpr int po1(void (*)() throws(M), T) { return 1; }
template <std::except_t M> constexpr int po1(void (*)() throws(M), int) { return 2; }
static_assert(po1(fs, 0) == 2);

// A noexcept(B) is not deduced from a dependent throws(M), so neither template
// is more specialized.
template <bool B, class T> constexpr int po2(void (*)() noexcept(B), T) { return 1; } // expected-note {{candidate function}}
template <std::except_t M> constexpr int po2(void (*)() throws(M), int) { return 2; } // expected-note {{candidate function}}
int p2 = po2(fn, 0); // expected-error {{call to 'po2' is ambiguous}}
