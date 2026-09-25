// P0709 4.5: try-expressions and catch sugars.
#include <error>
#include <iostream>
#include <string>
#include <climits>

bool flip = false;
bool flip_a_coin() { return flip; }

std::string f() throws {
  if (flip_a_coin()) throw std::errc::operation_not_permitted;
  return try "xyzzy" + std::string("plover");     // can grep for exception paths
}

std::string f2() throws {
  if (flip_a_coin()) throw std::errc::operation_not_permitted;
  try std::string s("xyzzy");                     // statement form
  try return s + "plover";
}

std::string g() throws { return try f() + "plugh"; }

int run() {
  auto result = try g();                          // can grep for exception paths
  std::cout << "success, result is: " << result << "\n";
  return 0;
  catch {                                         // clean syntax for the efficient catch
    std::cout << "failed, error is: " << err.message() << "\n";
    return 1;
  }
}

int safe_divide(int i, int j) throws {
  if (j == 0) throw std::errc::invalid_argument;
  if (i == INT_MIN && j == -1) throw std::errc::value_too_large;
  if (i % j != 0) throw std::errc::result_out_of_range;
  return i / j;
}

double caller(double i, int j, int k) throws { return i + try safe_divide(j, k); }

int caller2(int i, int j) {
  try return safe_divide(i, j);
  catch {
    if (err == std::errc::invalid_argument) return 0;
    if (err == std::errc::result_out_of_range) return i / j;
    if (err == std::errc::value_too_large) return INT_MIN;
    return -1;
  }
}

int main() {
  int failures = 0;
  if (run() != 0) ++failures;
  flip = true;
  if (run() != 1) ++failures;
  flip = false;
  if (f2() != "xyzzyplover") ++failures;
  if (caller2(9, 3) != 3 || caller2(9, 0) != 0 || caller2(9, 2) != 4 || caller2(INT_MIN, -1) != INT_MIN) ++failures;
  if (caller(0.5, 8, 2) != 4.5) ++failures;
  std::cout << (failures ? "FAILED" : "OK") << "\n";
  return failures;
}
