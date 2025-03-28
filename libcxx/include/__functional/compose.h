// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___FUNCTIONAL_COMPOSE_H
#define _LIBCPP___FUNCTIONAL_COMPOSE_H

#include <__config>
#include <__functional/invoke.h>
#include <__functional/perfect_forward.h>
#include <__utility/forward.h>
#include <type_traits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#  pragma clang include_instead(<functional>)
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER > 17

struct __compose_op {
    template<class _Fn1, class _Fn2, class ..._Args>
    _LIBCPP_HIDE_FROM_ABI
    constexpr auto operator()(_Fn1&& __f1, _Fn2&& __f2, _Args&&... __args) const
        noexcept(noexcept(_VSTD::invoke(_VSTD::forward<_Fn1>(__f1), _VSTD::invoke(_VSTD::forward<_Fn2>(__f2), _VSTD::forward<_Args>(__args)...))))
        -> decltype(      _VSTD::invoke(_VSTD::forward<_Fn1>(__f1), _VSTD::invoke(_VSTD::forward<_Fn2>(__f2), _VSTD::forward<_Args>(__args)...)))
        { return          _VSTD::invoke(_VSTD::forward<_Fn1>(__f1), _VSTD::invoke(_VSTD::forward<_Fn2>(__f2), _VSTD::forward<_Args>(__args)...)); }
};

template <class _Fn1, class _Fn2>
struct __compose_t : __perfect_forward<__compose_op, _Fn1, _Fn2> {
    using __perfect_forward<__compose_op, _Fn1, _Fn2>::__perfect_forward;
};

template <class _Fn1, class _Fn2>
_LIBCPP_HIDE_FROM_ABI
constexpr auto __compose(_Fn1&& __f1, _Fn2&& __f2)
    noexcept(noexcept(__compose_t<decay_t<_Fn1>, decay_t<_Fn2>>(_VSTD::forward<_Fn1>(__f1), _VSTD::forward<_Fn2>(__f2))))
    -> decltype(      __compose_t<decay_t<_Fn1>, decay_t<_Fn2>>(_VSTD::forward<_Fn1>(__f1), _VSTD::forward<_Fn2>(__f2)))
    { return          __compose_t<decay_t<_Fn1>, decay_t<_Fn2>>(_VSTD::forward<_Fn1>(__f1), _VSTD::forward<_Fn2>(__f2)); }

#endif // _LIBCPP_STD_VER > 17

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___FUNCTIONAL_COMPOSE_H
