// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___NUMERIC_INNER_PRODUCT_H
#define _LIBCPP___NUMERIC_INNER_PRODUCT_H

#include <__config>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#  pragma clang include_instead(<numeric>)
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _InputIterator1, class _InputIterator2, class _Tp>
_LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX17
_Tp
inner_product(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _Tp __init)
{
    for (; __first1 != __last1; ++__first1, (void) ++__first2)
#if _LIBCPP_STD_VER > 17
        __init = _VSTD::move(__init) + *__first1 * *__first2;
#else
        __init = __init + *__first1 * *__first2;
#endif
    return __init;
}

template <class _InputIterator1, class _InputIterator2, class _Tp, class _BinaryOperation1, class _BinaryOperation2>
_LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX17
_Tp
inner_product(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2,
              _Tp __init, _BinaryOperation1 __binary_op1, _BinaryOperation2 __binary_op2)
{
    for (; __first1 != __last1; ++__first1, (void) ++__first2)
#if _LIBCPP_STD_VER > 17
        __init = __binary_op1(_VSTD::move(__init), __binary_op2(*__first1, *__first2));
#else
        __init = __binary_op1(__init, __binary_op2(*__first1, *__first2));
#endif
    return __init;
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___NUMERIC_INNER_PRODUCT_H
