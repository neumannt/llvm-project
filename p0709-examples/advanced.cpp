#include <error>
#include <cstdio>
#include <cstdarg>
#include <string>
#include <stdexcept>
#include <memory>

int failures = 0;
#define CHECK(x) do { if (!(x)) { printf("FAIL line %d: %s\n", __LINE__, #x); ++failures; } } while (0)

int live = 0;
struct Obj {
  int v;
  Obj(int v) : v(v) { ++live; }
  Obj(const Obj &o) : v(o.v) { ++live; }
  ~Obj() { --live; }
};

bool fail = false;
void maybe() throws { if (fail) throw std::errc::io_error; }

// Class template with throws members.
template <class T> struct Box {
  T value;
  T get() const throws { maybe(); return value; }
  template <class U> U as() const throws { maybe(); return static_cast<U>(value); }
};

// Multiple inheritance: this-adjusting thunks for throws virtuals.
struct A { virtual int fa() throws { maybe(); return 1; } virtual ~A() = default; int a = 0; };
struct B { virtual int fb(int x) throws { maybe(); return x; } virtual ~B() = default; int b = 0; };
struct C : A, B {
  int fb(int x) throws override { maybe(); return x * 10; }
};

// Covariant return.
struct Base { virtual Base *clone() throws { maybe(); return this; } virtual ~Base() = default; };
struct Der : Base { int pad = 0; Der *clone() throws override { maybe(); return this; } };
struct Other { virtual ~Other() = default; int z = 0; };
struct Der2 : Other, Base { Der2 *clone() throws override { maybe(); return this; } };

// Variadic throws function.
int sum(int n, ...) throws {
  va_list ap;
  va_start(ap, n);
  int s = 0;
  for (int i = 0; i < n; ++i) s += va_arg(ap, int);
  va_end(ap);
  if (s < 0) throw std::errc::result_out_of_range;
  return s;
}

// sret return + NRVO.
std::string make_string(bool f) throws {
  std::string s = "hello, world, this string is long enough to allocate";
  if (f) throw std::errc::invalid_argument;
  return s;
}
Obj make_obj(bool f) throws {
  Obj o(5);
  if (f) throw std::errc::invalid_argument;
  return o;
}

// Temporaries inside a full-expression.
std::string concat(bool f) throws {
  return std::string("a") + make_string(false) + (f ? make_string(true) : std::string("b"));
}

// Partial array destruction.
Obj make_nth(int i) throws { if (i == 2) throw std::errc::bad_message; return Obj(i); }
int array_init() throws {
  Obj arr[4] = {make_nth(0), make_nth(1), make_nth(2), make_nth(3)};
  return arr[0].v;
}

// Static exception from inside a dynamic handler.
int from_handler() throws {
  try {
    throw std::runtime_error("dyn");
  } catch (const std::exception &) {
    Obj o(1);
    throw std::errc::operation_canceled;
  }
}

// break/continue/goto around static errors.
int loops() throws {
  int n = 0;
  for (int i = 0; i < 10; ++i) {
    Obj o(i);
    try {
      fail = (i % 3 == 0);
      maybe();
      if (i == 8) break;
      continue;
    } catch (std::error) {
      ++n;
      if (i == 6) goto done;
    }
  }
done:
  fail = false;
  return n;
}

// Throws returning a reference.
int global = 7;
int &ref(bool f) throws { if (f) throw std::errc::bad_address; return global; }

// Recursion.
int depth(int n) throws { if (n == 0) throw std::errc::no_buffer_space; Obj o(n); return depth(n - 1); }

// Constructor function-try-block rethrows implicitly.
struct FTB {
  Obj o;
  FTB() throws try : o(make_nth(2)) {
  } catch (std::error e) {
    CHECK(e == std::errc::bad_message);
  }
};

// Delegating constructor.
struct Deleg {
  Obj o{1};
  Deleg(int x) throws { if (x) throw std::errc::device_or_resource_busy; }
  Deleg() throws : Deleg(1) {}
};

// Member function pointers.
struct M { int f(int x) throws { if (x < 0) throw std::errc::invalid_argument; return x; } };

// unique_ptr with throws factory.
std::unique_ptr<Obj> factory(bool f) throws { auto p = std::make_unique<Obj>(3); maybe(); if (f) throw std::errc::timed_out; return p; }

int main() {
  Box<int> bi{41};
  CHECK(bi.get() == 41);
  CHECK(bi.as<long>() == 41);
  fail = true;
  try { bi.get(); CHECK(false); } catch (std::error e) { CHECK(e == std::errc::io_error); }
  fail = false;

  C c;
  B *pb = &c;
  CHECK(pb->fb(3) == 30);
  fail = true;
  try { pb->fb(3); CHECK(false); } catch (std::error e) { CHECK(e == std::errc::io_error); }
  fail = false;

  Der2 d2;
  Base *bp = &d2;
  CHECK(bp->clone() == static_cast<Base *>(&d2));
  fail = true;
  try { bp->clone(); CHECK(false); } catch (std::error) {}
  fail = false;

  CHECK(sum(3, 1, 2, 3) == 6);
  try { sum(2, -5, 1); CHECK(false); } catch (std::error e) { CHECK(e == std::errc::result_out_of_range); }

  CHECK(make_string(false).size() > 20);
  try { make_string(true); CHECK(false); } catch (std::error) {}
  { Obj o = make_obj(false); CHECK(o.v == 5 && live == 1); }
  CHECK(live == 0);
  try { Obj o = make_obj(true); CHECK(false); } catch (std::error) {}
  CHECK(live == 0);

  CHECK(concat(false).size() > 20);
  try { concat(true); CHECK(false); } catch (std::error) {}

  try { array_init(); CHECK(false); } catch (std::error e) { CHECK(e == std::errc::bad_message); }
  CHECK(live == 0);

  try { from_handler(); CHECK(false); } catch (std::error e) { CHECK(e == std::errc::operation_canceled); }
  CHECK(live == 0);

  CHECK(loops() == 3);
  CHECK(live == 0);

  CHECK(&ref(false) == &global);
  try { ref(true); CHECK(false); } catch (std::error e) { CHECK(e == std::errc::bad_address); }

  try { depth(10); CHECK(false); } catch (std::error e) { CHECK(e == std::errc::no_buffer_space); }
  CHECK(live == 0);

  try { FTB f; CHECK(false); } catch (std::error e) { CHECK(e == std::errc::bad_message); }
  CHECK(live == 0);

  try { Deleg d; CHECK(false); } catch (std::error e) { CHECK(e == std::errc::device_or_resource_busy); }
  CHECK(live == 0);

  M m;
  int (M::*mp)(int) throws = &M::f;
  CHECK((m.*mp)(4) == 4);
  try { (m.*mp)(-4); CHECK(false); } catch (std::error) {}

  CHECK(factory(false)->v == 3);
  CHECK(live == 0);
  try { factory(true); CHECK(false); } catch (std::error e) { CHECK(e == std::errc::timed_out); }
  CHECK(live == 0);

  printf("%s\n", failures ? "FAILED" : "OK");
  return failures != 0;
}
