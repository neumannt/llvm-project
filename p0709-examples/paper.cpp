// Examples from P0709R4 section 4.1.
#include <error>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <cctype>

enum class arithmetic_errc {
  divide_by_zero = 1,
  not_integer_division,
  integer_divide_overflows,
};

struct arithmetic_category_t : std::error_category {
  const char *name() const noexcept override { return "arithmetic"; }
  std::string message(int v) const override {
    switch (static_cast<arithmetic_errc>(v)) {
    case arithmetic_errc::divide_by_zero: return "divide by zero";
    case arithmetic_errc::not_integer_division: return "not integer division";
    case arithmetic_errc::integer_divide_overflows: return "integer divide overflows";
    }
    return "unknown";
  }
};
inline const arithmetic_category_t arithmetic_category{};
std::error_code make_error_code(arithmetic_errc e) {
  return {static_cast<int>(e), arithmetic_category};
}
template <> struct std::is_error_code_enum<arithmetic_errc> : std::true_type {};

int safe_divide(int i, int j) throws {
  if (j == 0)
    throw arithmetic_errc::divide_by_zero;
  if (i == INT_MIN && j == -1)
    throw arithmetic_errc::integer_divide_overflows;
  if (i % j != 0)
    throw arithmetic_errc::not_integer_division;
  else
    return i / j;
}

double caller(double i, double j, double k) throws {
  return i + safe_divide(j, k);
}

int caller2(int i, int j) noexcept {
  try {
    return safe_divide(i, j);
  } catch (std::error e) {
    if (e == arithmetic_errc::divide_by_zero)
      return 0;
    if (e == arithmetic_errc::not_integer_division)
      return i / j; // ignore
    if (e == arithmetic_errc::integer_divide_overflows)
      return INT_MIN;
  }
  return -1;
}

enum class ConversionErrc { EmptyString = 1, IllegalChar, TooLong };
struct conversion_category_t : std::error_category {
  const char *name() const noexcept override { return "conversion"; }
  std::string message(int v) const override { return "conversion error " + std::to_string(v); }
};
inline const conversion_category_t conversion_category{};
std::error_code make_error_code(ConversionErrc e) {
  return {static_cast<int>(e), conversion_category};
}
template <> struct std::is_error_code_enum<ConversionErrc> : std::true_type {};

int convert(const std::string &str) throws {
  if (str.empty())
    throw ConversionErrc::EmptyString;
  if (!std::all_of(str.begin(), str.end(), ::isdigit))
    throw ConversionErrc::IllegalChar;
  if (str.length() > 9)
    throw ConversionErrc::TooLong;
  return atoi(str.c_str());
}

int str_multiply(const std::string &s, int i) throws {
  auto result = convert(s);
  return result * i;
}

int failures = 0;
#define CHECK(x) do { if (!(x)) { printf("FAIL line %d: %s\n", __LINE__, #x); ++failures; } } while (0)

int main() {
  CHECK(caller2(10, 2) == 5);
  CHECK(caller2(10, 0) == 0);
  CHECK(caller2(10, 3) == 3);
  CHECK(caller2(INT_MIN, -1) == INT_MIN);

  try {
    double d = caller(1.0, 10, 5);
    CHECK(d == 3.0);
    caller(1.0, 10, 0);
    CHECK(false);
  } catch (std::error e) {
    CHECK(e == arithmetic_errc::divide_by_zero);
    printf("caught: %s\n", e.message().c_str());
  }

  try {
    CHECK(str_multiply("21", 2) == 42);
    str_multiply("x1", 2);
    CHECK(false);
  } catch (std::error e) {
    CHECK(e == ConversionErrc::IllegalChar);
    CHECK(e != ConversionErrc::TooLong);
  }

  // A static exception escaping into a function without 'throws' becomes a
  // dynamic exception.
  try {
    [] { str_multiply("", 1); }();
    CHECK(false);
  } catch (const std::error &e) {
    CHECK(e == ConversionErrc::EmptyString);
  }

  printf("%s\n", failures ? "FAILED" : "OK");
  return failures != 0;
}
