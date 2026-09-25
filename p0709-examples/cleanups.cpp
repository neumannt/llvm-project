// Unwinding semantics of static exceptions: all cleanups must run.
#include <error>
#include <cstdio>
#include <new>
#include <stdexcept>
#include <vector>
#include <string>

int failures = 0;
#define CHECK(x) do { if (!(x)) { printf("FAIL line %d: %s\n", __LINE__, #x); ++failures; } } while (0)

std::string trace;
struct Tracer {
  char c;
  Tracer(char c) : c(c) { trace += '+'; trace += c; }
  Tracer(const Tracer &o) : c(o.c) { trace += '+'; trace += c; }
  ~Tracer() { trace += '-'; trace += c; }
};

bool fail_now = false;
int may_fail(int v) throws {
  if (fail_now)
    throw std::errc::invalid_argument;
  return v;
}

// Locals and temporaries are destroyed.
int locals() throws {
  Tracer a('a');
  {
    Tracer b('b');
    Tracer(('t')), may_fail(1);
  }
  Tracer c('c');
  return may_fail(2);
}

// Partially constructed object: members constructed so far are destroyed.
struct Member {
  Tracer t{'m'};
  int x = may_fail(3);
  Tracer u{'n'};
};
struct Holder {
  Tracer h{'h'};
  Member m;
  Holder() throws {}
};

// Constructor that fails in a new-expression: memory is freed.
int live_objects = 0;
struct Heap {
  static void *operator new(size_t n) { ++live_objects; return ::operator new(n); }
  static void operator delete(void *p) { --live_objects; ::operator delete(p); }
  Tracer t{'x'};
  Heap() throws { may_fail(4); }
};

Heap *make_heap() throws { return new Heap(); }

// Loop with try inside, error slot must be reset each iteration.
int loop() throws {
  int caught = 0;
  for (int i = 0; i < 4; ++i) {
    try {
      fail_now = (i % 2) == 0;
      may_fail(i);
    } catch (std::error e) {
      ++caught;
    }
  }
  fail_now = false;
  return caught;
}

// Nested try: inner handler doesn't match statically (catch(int)), outer does.
int nested() throws {
  try {
    try {
      Tracer n('q');
      may_fail(5);
    } catch (int) {
      return -1;
    }
  } catch (const std::error &e) {
    return e == std::errc::invalid_argument ? 1 : -2;
  }
  return 0;
}

// Rethrow from a handler.
int rethrow() throws {
  try {
    may_fail(6);
  } catch (std::error e) {
    trace += "R";
    throw;
  }
  return 0;
}

// catch(...) catches static exceptions too; rethrow keeps them static.
int catch_all_rethrow() throws {
  try {
    may_fail(7);
  } catch (...) {
    trace += "A";
    throw;
  }
  return 0;
}

int main() {
  fail_now = true;
  trace.clear();
  try { locals(); CHECK(false); } catch (std::error e) { CHECK(e == std::errc::invalid_argument); }
  CHECK(trace == "+a+b+t-t-b-a");
  printf("locals: %s\n", trace.c_str());

  trace.clear();
  try { Holder h; CHECK(false); } catch (std::error) {}
  CHECK(trace == "+h+m-m-h");
  printf("members: %s\n", trace.c_str());

  trace.clear();
  try { make_heap(); CHECK(false); } catch (std::error) {}
  CHECK(trace == "+x-x");
  CHECK(live_objects == 0);
  printf("new: %s live=%d\n", trace.c_str(), live_objects);

  fail_now = false;
  trace.clear();
  CHECK(locals() == 2);
  CHECK(trace == "+a+b+t-t-b+c-c-a");
  {
    Heap *h = make_heap();
    CHECK(live_objects == 1);
    delete h;
    CHECK(live_objects == 0);
  }

  CHECK(loop() == 2);

  fail_now = true;
  trace.clear();
  CHECK(nested() == 1);
  CHECK(trace == "+q-q");

  trace.clear();
  try { rethrow(); CHECK(false); } catch (std::error e) { CHECK(e == std::errc::invalid_argument); }
  CHECK(trace == "R");
  trace.clear();
  try { catch_all_rethrow(); CHECK(false); } catch (std::error e) { CHECK(e == std::errc::invalid_argument); }
  CHECK(trace == "A");

  printf("%s\n", failures ? "FAILED" : "OK");
  return failures != 0;
}
