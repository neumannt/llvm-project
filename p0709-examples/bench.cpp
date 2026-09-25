// Error propagation through 3 levels of non-inlined calls:
// P0709 static exceptions vs. dynamic exceptions vs. hand-written error codes.
#include <error>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define NOINLINE __attribute__((noinline))
static std::vector<int> input;

// --- static exceptions
NOINLINE int s3(int x) throws { if (x < 0) throw std::errc::invalid_argument; return x + 1; }
NOINLINE int s2(int x) throws { return s3(x) * 2; }
NOINLINE int s1(int x) throws { return s2(x) + 3; }
NOINLINE long run_static() {
  long sum = 0;
  for (int x : input) {
    try { sum += s1(x); } catch (std::error e) { sum -= 1; }
  }
  return sum;
}

// --- dynamic exceptions
struct Err { int code; };
NOINLINE int d3(int x) { if (x < 0) throw Err{22}; return x + 1; }
NOINLINE int d2(int x) { return d3(x) * 2; }
NOINLINE int d1(int x) { return d2(x) + 3; }
NOINLINE long run_dynamic() {
  long sum = 0;
  for (int x : input) {
    try { sum += d1(x); } catch (const Err &) { sum -= 1; }
  }
  return sum;
}

// --- error codes (by hand)
NOINLINE int e3(int x, int &out) { if (x < 0) return 22; out = x + 1; return 0; }
NOINLINE int e2(int x, int &out) { int v; if (int ec = e3(x, v)) return ec; out = v * 2; return 0; }
NOINLINE int e1(int x, int &out) { int v; if (int ec = e2(x, v)) return ec; out = v + 3; return 0; }
NOINLINE long run_codes() {
  long sum = 0;
  for (int x : input) {
    int v;
    if (e1(x, v)) sum -= 1; else sum += v;
  }
  return sum;
}

template <class F> double time_it(F f, long &result) {
  auto t0 = std::chrono::steady_clock::now();
  result = f();
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::nano>(t1 - t0).count() / input.size();
}

int main() {
  const int N = 2000000;
  printf("%-10s %12s %12s %12s   (ns per call)\n", "failures", "throws", "throw", "error code");
  for (double rate : {0.0, 0.01, 0.1, 0.5}) {
    input.assign(N, 1);
    srand(42);
    for (int &x : input) x = (rand() < rate * RAND_MAX) ? -1 : rand() % 1000;
    long rs, rd, rc;
    double ts = 1e9, td = 1e9, tc = 1e9;
    for (int rep = 0; rep < 5; ++rep) {
      ts = std::min(ts, time_it(run_static, rs));
      td = std::min(td, time_it(run_dynamic, rd));
      tc = std::min(tc, time_it(run_codes, rc));
    }
    if (rs != rd || rs != rc) { printf("mismatch\n"); return 1; }
    printf("%9.0f%% %12.2f %12.2f %12.2f\n", rate * 100, ts, td, tc);
  }
}
