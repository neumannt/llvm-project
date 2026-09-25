// P0709 4.1.4: conditional throws and the throws operator.
#include <error>
#include <cstdio>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <algorithm>

int failures = 0;
#define CHECK(x) do { if (!(x)) { printf("FAIL line %d: %s\n", __LINE__, #x); ++failures; } } while (0)

int s(int x) throws { if (x < 0) throw std::errc::invalid_argument; return x; }
int d(int x) { if (x < 0) throw std::runtime_error("neg"); return x; }
int n(int x) noexcept { return x; }

static_assert(throws(n(1)) == std::no_except);
static_assert(throws(s(1)) == std::static_except);
static_assert(throws(d(1)) == std::dynamic_except);
static_assert(throws(s(1) + d(1)) == std::dynamic_except);
static_assert(throws(s(1) + n(1)) == std::static_except);
static_assert(std::is_same_v<decltype(throws(n(1))), std::except_t>);
// max() combines modes.
static_assert(std::max({throws(n(1)), throws(s(1))}) == std::static_except);

// Non-dependent conditions.
int c0(int x) throws(std::no_except) { return x; }
int c1(int x) throws(std::static_except) { if (x < 0) throw std::errc::io_error; return x; }
int c2(int x) throws(std::dynamic_except) { if (x < 0) throw std::logic_error("x"); return x; }
static_assert(noexcept(c0(1)));
static_assert(std::is_same_v<decltype(c1), int(int) throws>);
static_assert(std::is_same_v<decltype(c2), int(int)>);
static_assert(std::is_same_v<decltype(c0), int(int) noexcept>);

// The paper's example: transform reports errors exactly like op does.
template <class In, class Out, class Op>
Out transform(In first, In last, Out out, Op op) throws(throws(op(*first))) {
  for (; first != last; ++first, ++out)
    *out = op(*first);
  return out;
}

auto sl = [](int x) throws { return s(x); };
auto dl = [](int x) { return d(x); };
auto nl = [](int x) noexcept { return n(x); };
static_assert(std::is_same_v<decltype(transform<int *, int *, decltype(sl)>),
                             int *(int *, int *, int *, decltype(sl)) throws>);
static_assert(std::is_same_v<decltype(transform<int *, int *, decltype(dl)>),
                             int *(int *, int *, int *, decltype(dl))>);
static_assert(noexcept(transform((int *)nullptr, (int *)nullptr, (int *)nullptr, nl)));

// Generic wrapper whose move constructor adapts (throws(cond) in a class,
// parsed after the class is complete).
template <class T> struct Wrapper {
  T value;
  Wrapper(T v) : value(std::move(v)) {}
  Wrapper(Wrapper &&o) throws(!std::is_nothrow_move_constructible_v<T>)
      : value(std::move(o.value)) {}
};
struct Nothrow { Nothrow() = default; Nothrow(Nothrow &&) noexcept {} };
struct Throwing { Throwing() = default; Throwing(Throwing &&) {} };
static_assert(std::is_nothrow_move_constructible_v<Wrapper<Nothrow>>);
static_assert(!std::is_nothrow_move_constructible_v<Wrapper<Throwing>>);
static_assert(throws(Wrapper<Throwing>(std::declval<Wrapper<Throwing>>())) == std::static_except);

int main() {
  int in[3] = {1, -2, 3}, out[3] = {};
  try {
    transform(in, in + 3, out, sl);
    CHECK(false);
  } catch (std::error e) {
    CHECK(e == std::errc::invalid_argument);
    CHECK(out[0] == 1);
  }
  try {
    transform(in, in + 3, out, dl);
    CHECK(false);
  } catch (const std::runtime_error &) {
  }
  transform(in, in + 3, out, nl);
  CHECK(out[1] == -2);

  CHECK(c1(1) == 1);
  try { c1(-1); CHECK(false); } catch (std::error e) { CHECK(e == std::errc::io_error); }

  Wrapper<Throwing> w{Throwing()};
  Wrapper<Throwing> w2(std::move(w));
  (void)w2;

  printf("%s\n", failures ? "FAILED" : "OK");
  return failures != 0;
}
