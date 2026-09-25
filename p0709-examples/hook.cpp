// P0709 4.4: set_on_error_propagation().
#include <error>
#include <cstdio>

int propagations = 0;
std::on_error_propagation previous = nullptr;

int leaf(int x) throws { if (x < 0) throw std::errc::invalid_argument; return x; }
int mid(int x) throws { return leaf(x) + 1; }
int top(int x) throws { return mid(x) * 2; }

int main() {
  previous = std::set_on_error_propagation([](std::error e) noexcept {
    ++propagations;
    if (previous) previous(e);
  });
  int failures = 0;
  if (top(1) != 4) ++failures;
  if (propagations != 0) ++failures;
  try { top(-1); ++failures; } catch (std::error e) {}
  // leaf, mid and top each exit with the error.
  printf("propagations: %d\n", propagations);
  if (propagations != 3) ++failures;
  printf("%s\n", failures ? "FAILED" : "OK");
  return failures;
}
