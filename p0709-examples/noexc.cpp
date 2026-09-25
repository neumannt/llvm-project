// Static exceptions with dynamic exceptions disabled (-fno-exceptions).
#include <error>
#include <cstdio>

int failures = 0;
#define CHECK(x) do { if (!(x)) { printf("FAIL line %d: %s\n", __LINE__, #x); ++failures; } } while (0)

int dtors = 0;
struct D { ~D() { ++dtors; } };

int parse_digit(char c) throws {
  if (c < '0' || c > '9')
    throw std::errc::invalid_argument;
  return c - '0';
}

int parse(const char *s) throws {
  D d;
  int v = 0;
  for (; *s; ++s)
    v = v * 10 + parse_digit(*s);
  return v;
}

int parse_or(const char *s, int def) noexcept {
  try {
    return parse(s);
  } catch (std::error e) {
    return e == std::errc::invalid_argument ? def : -1;
  }
}

struct Base {
  virtual int get(int x) throws { if (x < 0) throw std::errc::result_out_of_range; return x; }
  virtual ~Base() = default;
};
struct Derived : Base {
  int get(int x) throws override { if (x > 100) throw std::errc::value_too_large; return Base::get(x) * 2; }
};

int via_pointer(int (*fp)(const char *) throws, const char *s) throws { return fp(s); }

int main() {
  CHECK(parse_or("123", 7) == 123);
  CHECK(parse_or("1x3", 7) == 7);
  CHECK(dtors == 2);

  Derived d;
  Base &b = d;
  try { CHECK(b.get(5) == 10); b.get(200); CHECK(false); }
  catch (std::error e) { CHECK(e == std::errc::value_too_large); }
  try { b.get(-1); CHECK(false); }
  catch (std::error e) { CHECK(e == std::errc::result_out_of_range); }

  try { CHECK(via_pointer(parse, "42") == 42); via_pointer(parse, "?"); CHECK(false); }
  catch (...) { printf("caught via catch(...)\n"); }

  auto l = [](int x) throws -> int { if (x) throw std::errc::io_error; return 1; };
  int (*lp)(int) throws = l;
  try { CHECK(lp(0) == 1); lp(1); CHECK(false); }
  catch (std::error e) { CHECK(e == std::errc::io_error); }

  printf("%s\n", failures ? "FAILED" : "OK");
  return failures != 0;
}
