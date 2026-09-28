// Compares the ABIs for 'throws' functions (-fstatic-exceptions-abi=pointer,
// register, carry): error propagation through chains of non-inlined calls,
// for several return types and failure rates. With -DDYNAMIC_EXCEPTIONS, the
// same code uses traditional (dynamic) exceptions as a baseline. Run
// abi-bench.sh to build and run all variants.
#ifdef DYNAMIC_EXCEPTIONS
#define THROWS
#define FAIL throw 22
#define CATCH catch (int)
#else
#include <error>
#define THROWS throws
#define FAIL throw std::errc::invalid_argument
#define CATCH catch (std::error)
#endif
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define NOINLINE __attribute__((noinline))
static std::vector<int> input;

// Keep the compiler from folding the calls away.
static volatile int sink;

// --- int through 8 levels
NOINLINE int i8(int x) THROWS { if (x < 0) FAIL; return x + 1; }
NOINLINE int i7(int x) THROWS { return i8(x) + 1; }
NOINLINE int i6(int x) THROWS { return i7(x) + 1; }
NOINLINE int i5(int x) THROWS { return i6(x) + 1; }
NOINLINE int i4(int x) THROWS { return i5(x) + 1; }
NOINLINE int i3(int x) THROWS { return i4(x) + 1; }
NOINLINE int i2(int x) THROWS { return i3(x) + 1; }
NOINLINE int i1(int x) THROWS { return i2(x) + 1; }
NOINLINE long run_int() {
  long sum = 0;
  for (int x : input) {
    try { sum += i1(x); } CATCH { sum -= 1; }
  }
  return sum;
}

// --- pure forwarding (tail calls), 8 levels
NOINLINE int t8(int x) THROWS { if (x < 0) FAIL; return x + 1; }
NOINLINE int t7(int x) THROWS { return t8(x + 1); }
NOINLINE int t6(int x) THROWS { return t7(x + 1); }
NOINLINE int t5(int x) THROWS { return t6(x + 1); }
NOINLINE int t4(int x) THROWS { return t5(x + 1); }
NOINLINE int t3(int x) THROWS { return t4(x + 1); }
NOINLINE int t2(int x) THROWS { return t3(x + 1); }
NOINLINE int t1(int x) THROWS { return t2(x < 0 ? x - 100 : x); }
NOINLINE long run_tail() {
  long sum = 0;
  for (int x : input) {
    try { sum += t1(x); } CATCH { sum -= 1; }
  }
  return sum;
}

// --- void through 4 levels, with work in between
NOINLINE void v4(int x) THROWS { if (x < 0) FAIL; sink = x; }
NOINLINE void v3(int x) THROWS { v4(x); sink = x + 1; }
NOINLINE void v2(int x) THROWS { v3(x); sink = x + 2; }
NOINLINE void v1(int x) THROWS { v2(x); sink = x + 3; }
NOINLINE long run_void() {
  long sum = 0;
  for (int x : input) {
    try { v1(x); sum += x; } CATCH { sum -= 1; }
  }
  return sum;
}

// --- double through 4 levels
NOINLINE double d4(int x) THROWS { if (x < 0) FAIL; return x * 0.5; }
NOINLINE double d3(int x) THROWS { return d4(x) + 1; }
NOINLINE double d2(int x) THROWS { return d3(x) + 1; }
NOINLINE double d1(int x) THROWS { return d2(x) + 1; }
NOINLINE long run_double() {
  long sum = 0;
  for (int x : input) {
    try { sum += (long)d1(x); } CATCH { sum -= 1; }
  }
  return sum;
}

// --- 16-byte struct (two registers) through 4 levels
struct Pair { long a, b; };
NOINLINE Pair p4(int x) THROWS { if (x < 0) FAIL; return {x, x + 1}; }
NOINLINE Pair p3(int x) THROWS { Pair p = p4(x); return {p.a + 1, p.b}; }
NOINLINE Pair p2(int x) THROWS { Pair p = p3(x); return {p.a, p.b + 1}; }
NOINLINE Pair p1(int x) THROWS { Pair p = p2(x); return {p.a + 1, p.b}; }
NOINLINE long run_pair() {
  long sum = 0;
  for (int x : input) {
    try { Pair p = p1(x); sum += p.a + p.b; } CATCH { sum -= 1; }
  }
  return sum;
}

// --- 32-byte struct (returned in memory) through 4 levels
struct Big { long a, b, c, d; };
NOINLINE Big b4(int x) THROWS { if (x < 0) FAIL; return {x, x, x, x}; }
NOINLINE Big b3(int x) THROWS { Big b = b4(x); b.a += 1; return b; }
NOINLINE Big b2(int x) THROWS { Big b = b3(x); b.b += 1; return b; }
NOINLINE Big b1(int x) THROWS { Big b = b2(x); b.c += 1; return b; }
NOINLINE long run_big() {
  long sum = 0;
  for (int x : input) {
    try { Big b = b1(x); sum += b.a + b.b + b.c + b.d; } CATCH { sum -= 1; }
  }
  return sum;
}

// --- pointer, leaf only, in a hot loop
static int table[1024];
NOINLINE int *lookup(int x) THROWS { if (x < 0) FAIL; return &table[x & 1023]; }
NOINLINE long run_ptr() {
  long sum = 0;
  for (int x : input) {
    try { sum += *lookup(x) + 1; } CATCH { sum -= 1; }
  }
  return sum;
}

template <class F> double time_it(F f, long &result) {
  auto t0 = std::chrono::steady_clock::now();
  result = f();
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::nano>(t1 - t0).count() / input.size();
}

int main(int argc, char **argv) {
  const int N = 2000000;
  struct { const char *name; long (*fn)(); } tests[] = {
      {"int x8", run_int},       {"tail x8", run_tail}, {"void x4", run_void},
      {"double x4", run_double}, {"pair x4", run_pair}, {"big x4", run_big},
      {"ptr x1", run_ptr},
  };
  printf("%-10s", "failures");
  for (auto &t : tests) printf(" %10s", t.name);
  printf("   (ns per outer call, %s)\n", argc > 1 ? argv[1] : "");
  unsigned long checksum = 0;
  for (double rate : {0.0, 0.01, 0.1, 0.5}) {
    input.assign(N, 1);
    srand(42);
    for (int &x : input) x = (rand() < rate * RAND_MAX) ? -1 : rand() % 1000;
    printf("%9.0f%%", rate * 100);
    for (auto &t : tests) {
      long r;
      double best = 1e9;
      for (int rep = 0; rep < 7; ++rep) best = std::min(best, time_it(t.fn, r));
      checksum = checksum * 31 + (unsigned long)r;
      printf(" %10.2f", best);
      fflush(stdout);
    }
    printf("\n");
  }
  // Must be the same for all ABIs.
  printf("checksum %lx\n", checksum);
}
