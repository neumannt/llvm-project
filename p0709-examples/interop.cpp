// Translation between dynamic and static exceptions.
#include <error>
#include <cstdio>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

int failures = 0;
#define CHECK(x) do { if (!(x)) { printf("FAIL line %d: %s\n", __LINE__, #x); ++failures; } } while (0)

struct MyException : std::runtime_error {
  int payload;
  MyException(int p) : std::runtime_error("mine"), payload(p) {}
};

[[noreturn]] void throw_bad_alloc() { throw std::bad_alloc(); }
[[noreturn]] void throw_mine(int p) { throw MyException(p); }
[[noreturn]] void throw_invalid() { throw std::invalid_argument("bad arg"); }

// Dynamic exceptions escaping a 'throws' function are translated.
int f_alloc() throws { throw_bad_alloc(); }
int f_mine(int p) throws { throw_mine(p); }
int f_invalid() throws { throw_invalid(); }
int f_dynamic_error() throws { throw_invalid(); }

// A dynamic throw statement inside a 'throws' function.
int f_throw_stmt() throws { throw std::out_of_range("oor"); }

// Dynamic exceptions caught locally are not translated.
int f_local() throws {
  try {
    throw_mine(3);
  } catch (const MyException &e) {
    return e.payload;
  }
}

// Calling a 'throws' function from a normal function: errors become dynamic
// exceptions (ENOMEM -> bad_alloc, wrapped exceptions are rethrown).
int g_alloc() { return f_alloc(); }
int g_mine() { return f_mine(42); }

int main() {
  try { f_alloc(); CHECK(false); }
  catch (std::error e) { CHECK(e == std::errc::not_enough_memory); }

  try { f_mine(1); CHECK(false); }
  catch (std::error e) {
    CHECK(e.is_dynamic_exception());
    printf("wrapped: %s\n", e.message().c_str());
    try { std::rethrow_exception(e.exception()); }
    catch (const MyException &m) { CHECK(m.payload == 1); }
  }

  try { f_invalid(); CHECK(false); }
  catch (std::error e) { CHECK(e == std::errc::invalid_argument); }

  try { f_throw_stmt(); CHECK(false); }
  catch (std::error e) { CHECK(e == std::errc::result_out_of_range); }

  CHECK(f_local() == 3);

  try { g_alloc(); CHECK(false); }
  catch (const std::bad_alloc &) { printf("bad_alloc as expected\n"); }

  try { g_mine(); CHECK(false); }
  catch (const MyException &m) { CHECK(m.payload == 42); }

  // catch(error) and catch(std::exception) side by side.
  int which = 0;
  try { f_alloc(); } catch (std::error) { which = 1; } catch (const std::exception &) { which = 2; }
  CHECK(which == 1);
  try { g_alloc(); } catch (std::error) { which = 3; } catch (const std::exception &) { which = 4; }
  CHECK(which == 4);
  try { (void)f_alloc(); } catch (...) { which = 5; }
  CHECK(which == 5);

  // Lambda declared throws translates.
  try {
    []() throws { std::vector<int> v; v.at(3); }();
    CHECK(false);
  } catch (std::error e) {
    printf("lambda: %s\n", e.message().c_str());
    CHECK(e == std::errc::result_out_of_range);
  }

  printf("%s\n", failures ? "FAILED" : "OK");
  return failures != 0;
}
