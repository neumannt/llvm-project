// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_FIND_FIRST_OF_H
#define _LIBCPP___ALGORITHM_FIND_FIRST_OF_H

#include <__algorithm/comp.h>
#include <__config>
#include <__iterator/iterator_traits.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#  pragma clang include_instead(<algorithm>)
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
_LIBCPP_CONSTEXPR_AFTER_CXX11 _ForwardIterator1 __find_first_of_ce(_ForwardIterator1 __first1,
                                                                   _ForwardIterator1 __last1,
                                                                   _ForwardIterator2 __first2,
                                                                   _ForwardIterator2 __last2, _BinaryPredicate __pred) {
  for (; __first1 != __last1; ++__first1)
    for (_ForwardIterator2 __j = __first2; __j != __last2; ++__j)
      if (__pred(*__first1, *__j))
        return __first1;
  return __last1;
}

template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
_LIBCPP_NODISCARD_EXT inline _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX17 _ForwardIterator1
find_first_of(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
              _ForwardIterator2 __last2, _BinaryPredicate __pred) {
  return _VSTD::__find_first_of_ce(__first1, __last1, __first2, __last2, __pred);
}

template <class _ForwardIterator1, class _ForwardIterator2>
_LIBCPP_NODISCARD_EXT inline _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX17 _ForwardIterator1 find_first_of(
    _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2) {
  typedef typename iterator_traits<_ForwardIterator1>::value_type __v1;
  typedef typename iterator_traits<_ForwardIterator2>::value_type __v2;
  return _VSTD::__find_first_of_ce(__first1, __last1, __first2, __last2, __equal_to<__v1, __v2>());
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_FIND_FIRST_OF_H
