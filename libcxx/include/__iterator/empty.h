// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ITERATOR_EMPTY_H
#define _LIBCPP___ITERATOR_EMPTY_H

#include <__config>
#include <cstddef>
#include <initializer_list>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#  pragma clang include_instead(<iterator>)
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER > 14

template <class _Cont>
_LIBCPP_NODISCARD_AFTER_CXX17 _LIBCPP_INLINE_VISIBILITY
constexpr auto empty(const _Cont& __c)
_NOEXCEPT_(noexcept(__c.empty()))
-> decltype        (__c.empty())
{ return            __c.empty(); }

template <class _Tp, size_t _Sz>
_LIBCPP_NODISCARD_AFTER_CXX17 _LIBCPP_INLINE_VISIBILITY
constexpr bool empty(const _Tp (&)[_Sz]) noexcept { return false; }

template <class _Ep>
_LIBCPP_NODISCARD_AFTER_CXX17 _LIBCPP_INLINE_VISIBILITY
constexpr bool empty(initializer_list<_Ep> __il) noexcept { return __il.size() == 0; }

#endif // _LIBCPP_STD_VER > 14

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ITERATOR_EMPTY_H
