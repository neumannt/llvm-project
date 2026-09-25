#include <error>
#include <cstdio>
#include <functional>
#include <vector>
#include <map>
#include <string>

int failures = 0;
#define CHECK(x) do { if (!(x)) { printf("FAIL line %d: %s\n", __LINE__, #x); ++failures; } } while (0)

bool fail = false;
struct Res {
  int v;
  Res(int v) throws : v(v) { if (fail) throw std::errc::no_space_on_device; }
  Res(const Res &o) throws : v(o.v) { if (fail) throw std::errc::no_space_on_device; }
};

// Inheriting constructor.
struct Derived : Res { using Res::Res; };

// Implicit copy constructor calling a 'throws' copy constructor: it is not
// 'throws' itself, so the failure becomes a dynamic exception.
struct Holder { Res r{1}; std::string s = "x"; };

// constexpr 'throws' function.
constexpr int cx(int x) throws { if (x < 0) throw std::errc::invalid_argument; return x * 2; }
static_assert(cx(21) == 42);

int main() {
  try { Derived d(5); CHECK(d.v == 5); fail = true; Derived e(6); CHECK(false); }
  catch (std::error e) { CHECK(e == std::errc::no_space_on_device); }
  fail = false;

  Holder h;
  fail = true;
  try { Holder h2 = h; CHECK(false); }
  catch (std::error e) { CHECK(e == std::errc::no_space_on_device); }
  fail = false;

  // std::function around a 'throws' lambda: std::function's call operator is
  // not 'throws', so the error arrives as a dynamic std::error.
  std::function<int(int)> fn = [](int x) throws { if (x) throw std::errc::io_error; return 7; };
  CHECK(fn(0) == 7);
  try { fn(1); CHECK(false); } catch (const std::error &e) { CHECK(e == std::errc::io_error); }

  // Containers of types with 'throws' copy constructors.
  std::vector<Res> v;
  for (int i = 0; i < 10; ++i) v.push_back(Res(i));
  fail = true;
  try { v.push_back(v[0]); CHECK(false); } catch (std::error e) {}
  fail = false;
  CHECK(v.size() == 10);

  std::map<std::string, int> m;
  auto lookup = [&](const std::string &k) throws -> int {
    auto it = m.find(k);
    if (it == m.end()) throw std::errc::no_such_file_or_directory;
    return it->second;
  };
  m["a"] = 1;
  CHECK(lookup("a") == 1);
  try { lookup("b"); CHECK(false); } catch (std::error e) { CHECK(e == std::errc::no_such_file_or_directory); }

  CHECK(cx(4) == 8);
  try { (void)cx(-4); CHECK(false); } catch (std::error) {}

  printf("%s\n", failures ? "FAILED" : "OK");
  return failures != 0;
}
