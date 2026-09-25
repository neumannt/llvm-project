// RUN: %clang_cc1 -std=c++20 -fstatic-exceptions -fcxx-exceptions -fexceptions -fblocks -triple x86_64-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++14 -fstatic-exceptions -triple x86_64-linux-gnu -fsyntax-only -verify=cxx14 -DCXX14 %s
// RUN: %clang_cc1 -std=c++17 -fstatic-exceptions -triple x86_64-windows-msvc -fsyntax-only -verify=msvc -DCXX14 %s

// Restrictions of the P0709 static exceptions prototype.

#ifdef CXX14
int f() throws; // cxx14-error {{'throws' requires C++17 or later}} \
                // msvc-error {{'throws' is not supported for this C++ ABI}}
#else
namespace std {
struct error { const void *domain; long value; };
error __error_from_current_exception() noexcept;
[[noreturn]] void __throw_error_as_dynamic(error);
template <class R, class... A> struct coroutine_traits {
  using promise_type = typename R::promise_type;
};
template <class P = void> struct coroutine_handle {
  static coroutine_handle from_address(void *);
};
} // namespace std

struct suspend_never {
  bool await_ready() noexcept { return true; }
  void await_suspend(std::coroutine_handle<>) noexcept {}
  void await_resume() noexcept {}
};
struct task {
  struct promise_type {
    task get_return_object();
    suspend_never initial_suspend();
    suspend_never final_suspend() noexcept;
    void return_void();
    void unhandled_exception();
  };
};

task coro() throws { co_return; } // expected-error {{coroutine cannot be declared 'throws'}}
int main() throws { return 0; } // expected-error {{'main' cannot be declared 'throws'}}
struct S {
  void operator delete(void *) throws; // expected-error {{deallocation function cannot be declared 'throws'}}
};
void blk() {
  auto b = ^() throws {}; // expected-error {{block cannot be declared 'throws'}}
}
#endif
