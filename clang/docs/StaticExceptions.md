# Static Exceptions (P0709 prototype)

```{contents}
:local: true
```

## Introduction

Clang contains an experimental prototype of
[P0709R4 "Zero-overhead deterministic exceptions: Throwing values"](https://wg21.link/p0709r4)
by Herb Sutter. It is enabled with `-fstatic-exceptions` (C++17 or later,
Itanium C++ ABI targets only).

A function declared with the static-exception-specification `throws` reports
failure by throwing a value of the single, statically known type `std::error`.
Throwing and propagating such an exception is implemented exactly like
returning an error code: there are no unwind tables, no dynamic allocation and
no RTTI involved, and the cost is deterministic.

```c++
#include <error>

int safe_divide(int i, int j) throws {
  if (j == 0)
    throw std::errc::invalid_argument;      // converted to std::error
  return i / j;
}

double caller(double i, int j, int k) throws {
  return i + safe_divide(j, k);             // errors propagate automatically
}

int caller2(int i, int j) noexcept {
  try {
    return safe_divide(i, j);
  } catch (std::error e) {                  // catch by value
    if (e == std::errc::invalid_argument)   // semantic comparison
      return 0;
    return -1;
  }
}
```

With optimizations enabled, `caller2` above compiles to a plain comparison
and branch; there is no landing pad and no call into the C++ runtime.

## The `<error>` header and `std::error`

Clang ships the header `<error>`, which defines:

- `std::error`: a `[[clang::trivial_abi]]` class of two pointers, which is
  passed and returned in registers and is trivially relocatable. It holds a
  pointer to a `std::error_category` (the *domain*, never null) and a value in
  that domain. An error that wraps a dynamic exception owns it; copies share
  it. It is constructible from `std::errc`, `std::error_code`,
  `std::error_condition` and error code/condition enums (via
  `make_error_code` / `make_error_condition`), and a default-constructed error
  is a nonspecific failure. `operator==` performs semantic comparison across
  domains using `error_category::equivalent`, so for example an `error_code`
  from `std::system_category()` compares equal to the corresponding `errc`
  value. `message()` returns a description.
- `std::except_t` with the enumerators `no_except`, `static_except` and
  `dynamic_except` (see below).
- `std::set_on_error_propagation` (see below).
- The helpers used by the compiler, `std::__error_from_current_exception` and
  `std::__throw_error_as_dynamic`.

The compiler looks these up by name. It requires `std::error` to be trivially
copyable or `[[clang::trivial_abi]]` (it moves errors with `memcpy`) and its
first non-static data member to be the domain pointer. After moving an error
with `memcpy`, the compiler may set the domain pointer of the source to null
instead of destroying it; the destructor must do nothing in that case.

## Semantics

### `throws` functions

- `throws` is part of the function type: `int() throws` and `int()` are
  different types, and function pointers do not convert between them. It is
  printed and mangled (as the vendor qualifier `U6throws`) accordingly.
- All declarations of a function must agree on `throws`, and a virtual
  function and its overriders must agree on it.
- For the `noexcept` operator and the `noexcept`/`nothrow` traits, a `throws`
  function behaves like `noexcept(false)`.
- Destructors, deallocation functions, `main`, coroutines and blocks cannot be
  declared `throws`.
- Lambdas can be declared `throws`: `[]() throws { ... }`. Their conversion to
  a function pointer yields a pointer to a `throws` function.

### Throwing

- In a `throws` function, `throw expr` throws a static exception if `expr` can
  be converted to `std::error`. Otherwise it throws a dynamic exception (which
  is then translated, see below). With `-Wstatic-exceptions-translation` the
  compiler warns about such dynamic throws.
- A static exception goes to the innermost enclosing local handler that is
  `catch (std::error)` (by value or reference) or `catch (...)`, as if by a
  forward `goto`. If there is none, the function returns the error to its
  caller. All destructors of local and temporary objects, of fully constructed
  subobjects, and the deallocation in a failing new-expression run, as for a
  dynamic exception.
- In a handler that caught a `std::error`, `throw;` rethrows the caught error
  (the catch parameter, or the object it refers to). In a `catch (...)`
  handler, `throw;` rethrows the exception that was caught, statically or
  dynamically. A rethrown static exception is moved, not copied: the
  destructors of the objects in the handler run first, with the error still
  intact, and then the error is moved to its target where it would have been
  destroyed. (If the target is a dynamic exception, the error is copied,
  which shares a wrapped dynamic exception.)

### Interaction with dynamic exceptions

- A dynamic exception that escapes the body of a `throws` function (including
  its constructor initializers) is translated to a `std::error`:
  a `std::error` thrown dynamically is returned as is, `std::bad_alloc`
  becomes `errc::not_enough_memory`, `std::system_error` becomes its error
  code, and any other exception is wrapped (the error then refers to an
  `exception_ptr`, see `std::error::exception()`; standard exception types
  compare equal to the corresponding `errc` values).
- When a static exception reaches a function that is not declared `throws`
  and is not handled there, it is thrown as a dynamic exception: a wrapped
  exception is rethrown unchanged, `errc::not_enough_memory` becomes
  `std::bad_alloc`, and any other error is thrown as a `std::error` object,
  which `catch (std::error)` can catch.
- `catch (std::error)` and `catch (...)` handle both static exceptions and
  (dynamically thrown) `std::error` objects; other handlers only see dynamic
  exceptions.

### Static exceptions without dynamic exceptions

With `-fstatic-exceptions -fno-exceptions`, `try`, `catch` and `throw` can
still be used for static exceptions. Dynamic exceptions remain unavailable;
a static exception that reaches a function that is not `throws` (and is not
handled there) aborts the program.

### Conditional `throws` and the `throws` operator

`throws(cond)` is a conditional static exception specification. `cond` is a
constant expression whose value selects the error reporting of the function:
`std::no_except` (0, like `noexcept`), `std::static_except` (1, like `throws`)
or `std::dynamic_except` (2, like no exception specification).

The operator `throws(expr)` yields the `std::except_t` of an unevaluated
expression: `no_except` if it cannot throw, otherwise `static_except` if it can
only throw static exceptions, otherwise `dynamic_except`. Together they let
generic code report errors exactly like the operations it calls:

```c++
template <class In, class Out, class Op>
Out transform(In first, In last, Out out, Op op) throws(throws(op(*first)));
```

Conditional `throws` specifications on member functions are parsed after the
class is complete, like `noexcept(expr)`. They are not allowed on virtual
functions.

### `try` expressions and `catch` shorthands (P0709 4.5)

- `try` can precede an expression, a subexpression, or a statement that is not
  a compound statement, to make exceptional control flow visible. It has no
  semantic effect:

  ```c++
  std::string g() throws { return try f() + "plugh"; }
  double caller(double i, int j, int k) throws { return i + try safe_divide(j, k); }
  try return s + "plover";
  ```

- `catch { ... }` is shorthand for `catch (std::error err) { ... }`.
- A `catch` that is not preceded by `try` handles everything from the
  enclosing `{` up to the `catch`. It must be at the end of its block, and
  the declarations before it are not visible in the handler:

  ```c++
  int main() {
    auto result = try g();
    std::cout << "success, result is: " << result;
    return 0;
    catch {
      std::cout << "failed, error is: " << err.message();
      return 1;
    }
  }
  ```

### Error propagation hook (P0709 4.4)

With `-fstatic-exceptions-propagation-hook`, every exit of a `throws` function
with an error calls the function installed with
`std::set_on_error_propagation(on_error_propagation)`, where
`on_error_propagation` is `void (*)(std::error) noexcept`. Each function has a
single call site for this. Without the flag there is no overhead.

## Implementation notes

- ABI: P0709 lists two strategies to prototype, and
  `-fstatic-exceptions-abi=` selects between them. All translation units of a
  program must use the same setting; it is not part of the mangling.

  - `pointer`: a `throws` function takes a hidden parameter, a
    pointer to a caller-owned `std::error` object, which follows the fixed
    parameters. To fail, the function stores the error there and returns (its
    return value is then unspecified). The caller tests the domain pointer
    after the call. A `throws` function forwards its own error pointer to the
    `throws` functions it calls, so propagation needs no copying.
  - `register` (the default) and `carry`: a `throws` function returns either its value or
    the error in registers, plus a flag that tells which. The integer and
    pointer parts of the return value share the first integer return
    registers with the error (a union); floating-point and vector parts keep
    their registers, and values returned in memory still are (through a
    pointer that is not marked `sret`, as the return registers are in use).
    For example, on x86-64 `int f() throws` returns the value or the error in
    RAX:RDX and `double g() throws` returns the value in XMM0 or the error in
    RAX:RDX. With `register`, the flag is returned in the next integer return
    register (RCX on x86-64, X2 on AArch64); the caller tests it after the
    call. With `carry` (x86 only), the flag is returned in the carry flag, as
    suggested by P0709: the callee returns with `clc` or `stc` and the caller
    branches with `jc`. Forwarding the result of a `throws` call remains a
    tail call. In the IR, the function returns a struct whose last element is
    the `i1` flag; `carry` adds the `"x86-carry-flag-return"` function
    attribute, which tells the x86 backend to return that element in CF.
    Right before instruction selection, the x86 backend duplicates the
    (merged) return blocks of such functions so that the flag is a constant
    for each `ret` where possible. `carry` is an experiment and not supported
    where the epilogue clobbers EFLAGS: on Win64 (which must deallocate the
    stack with `add` when there is no frame pointer) and with
    `-fzero-call-used-regs` (which clears registers with `xor`).

  With every ABI, `return f(...);` in a `throws` function, where `f` is a
  `throws` function with the same return type, returns the result of `f`
  as is (success or error) if no cleanups have to run and no local handler or
  propagation hook is involved, so that the call becomes a tail call.
  `p0709-examples/abi-bench.sh` compares the three ABIs.
- `throws` functions and calls to them are `nounwind`. Dynamic exceptions are
  caught by an implicit catch-all around the function body, which is only
  emitted if something in the function can throw dynamically.
- The static exception path does not use the EH machinery: cleanups are
  emitted inline on the branch to the target handler or return.

## Limitations of the prototype

- Only the Itanium C++ ABI with landing-pad based EH is supported (not MSVC,
  WebAssembly or other funclet-based EH).
- With `-fstatic-exceptions-abi=register` or `carry`, error returns bypass
  the function exit instrumentation (`-finstrument-functions`), and
  `std::error` must be two words in size. Virtual member function pointer
  thunks (with pointer authentication) of `throws` functions are not
  supported.
- `throws` functions are `nounwind`, so forced unwinding (e.g. thread
  cancellation with `pthread_exit` or `pthread_cancel`) cannot pass through
  them, just like through `noexcept` functions: it aborts the program.
- `throw;` refers to the static exception only if it appears lexically in the
  handler (not in a function called from the handler, or in a lambda).
- `std::error` is not yet an evolution of `std::error_code` as in P1028; it
  reuses `std::error_category` for its domains.
- The "treat every function as `throws`" migration mode, overloading on
  `throws`, `throws{E}` with other error types and catching by value
  (`catch (errc::x)`) are not implemented.
