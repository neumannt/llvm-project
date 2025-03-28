//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_REVERSE_H
#define _LIBCPP___ALGORITHM_REVERSE_H

#include <__algorithm/iter_swap.h>
#include <__config>
#include <__iterator/iterator_traits.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#  pragma clang include_instead(<algorithm>)
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _BidirectionalIterator>
inline _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX17
void
__reverse(_BidirectionalIterator __first, _BidirectionalIterator __last, bidirectional_iterator_tag)
{
    while (__first != __last)
    {
        if (__first == --__last)
            break;
        _VSTD::iter_swap(__first, __last);
        ++__first;
    }
}

template <class _RandomAccessIterator>
inline _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX17
void
__reverse(_RandomAccessIterator __first, _RandomAccessIterator __last, random_access_iterator_tag)
{
    if (__first != __last)
        for (; __first < --__last; ++__first)
            _VSTD::iter_swap(__first, __last);
}

template <class _BidirectionalIterator>
inline _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX17
void
reverse(_BidirectionalIterator __first, _BidirectionalIterator __last)
{
    _VSTD::__reverse(__first, __last, typename iterator_traits<_BidirectionalIterator>::iterator_category());
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_REVERSE_H
