//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_STABLE_PARTITION_H
#define _LIBCPP___ALGORITHM_STABLE_PARTITION_H

#include <__algorithm/rotate.h>
#include <__config>
#include <__iterator/iterator_traits.h>
#include <__utility/swap.h>
#include <memory>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#  pragma clang include_instead(<algorithm>)
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Predicate, class _ForwardIterator, class _Distance, class _Pair>
_ForwardIterator
__stable_partition(_ForwardIterator __first, _ForwardIterator __last, _Predicate __pred,
                   _Distance __len, _Pair __p, forward_iterator_tag __fit)
{
    // *__first is known to be false
    // __len >= 1
    if (__len == 1)
        return __first;
    if (__len == 2)
    {
        _ForwardIterator __m = __first;
        if (__pred(*++__m))
        {
            swap(*__first, *__m);
            return __m;
        }
        return __first;
    }
    if (__len <= __p.second)
    {   // The buffer is big enough to use
        typedef typename iterator_traits<_ForwardIterator>::value_type value_type;
        __destruct_n __d(0);
        unique_ptr<value_type, __destruct_n&> __h(__p.first, __d);
        // Move the falses into the temporary buffer, and the trues to the front of the line
        // Update __first to always point to the end of the trues
        value_type* __t = __p.first;
        ::new ((void*)__t) value_type(_VSTD::move(*__first));
        __d.template __incr<value_type>();
        ++__t;
        _ForwardIterator __i = __first;
        while (++__i != __last)
        {
            if (__pred(*__i))
            {
                *__first = _VSTD::move(*__i);
                ++__first;
            }
            else
            {
                ::new ((void*)__t) value_type(_VSTD::move(*__i));
                __d.template __incr<value_type>();
                ++__t;
            }
        }
        // All trues now at start of range, all falses in buffer
        // Move falses back into range, but don't mess up __first which points to first false
        __i = __first;
        for (value_type* __t2 = __p.first; __t2 < __t; ++__t2, (void) ++__i)
            *__i = _VSTD::move(*__t2);
        // __h destructs moved-from values out of the temp buffer, but doesn't deallocate buffer
        return __first;
    }
    // Else not enough buffer, do in place
    // __len >= 3
    _ForwardIterator __m = __first;
    _Distance __len2 = __len / 2;  // __len2 >= 2
    _VSTD::advance(__m, __len2);
    // recurse on [__first, __m), *__first know to be false
    // F?????????????????
    // f       m         l
    _ForwardIterator __first_false = _VSTD::__stable_partition<_Predicate&>(__first, __m, __pred, __len2, __p, __fit);
    // TTTFFFFF??????????
    // f  ff   m         l
    // recurse on [__m, __last], except increase __m until *(__m) is false, *__last know to be true
    _ForwardIterator __m1 = __m;
    _ForwardIterator __second_false = __last;
    _Distance __len_half = __len - __len2;
    while (__pred(*__m1))
    {
        if (++__m1 == __last)
            goto __second_half_done;
        --__len_half;
    }
    // TTTFFFFFTTTF??????
    // f  ff   m  m1     l
    __second_false = _VSTD::__stable_partition<_Predicate&>(__m1, __last, __pred, __len_half, __p, __fit);
__second_half_done:
    // TTTFFFFFTTTTTFFFFF
    // f  ff   m    sf   l
    return _VSTD::rotate(__first_false, __m, __second_false);
    // TTTTTTTTFFFFFFFFFF
    //         |
}

template <class _Predicate, class _ForwardIterator>
_ForwardIterator
__stable_partition(_ForwardIterator __first, _ForwardIterator __last, _Predicate __pred,
                   forward_iterator_tag)
{
    const unsigned __alloc_limit = 3;  // might want to make this a function of trivial assignment
    // Either prove all true and return __first or point to first false
    while (true)
    {
        if (__first == __last)
            return __first;
        if (!__pred(*__first))
            break;
        ++__first;
    }
    // We now have a reduced range [__first, __last)
    // *__first is known to be false
    typedef typename iterator_traits<_ForwardIterator>::difference_type difference_type;
    typedef typename iterator_traits<_ForwardIterator>::value_type value_type;
    difference_type __len = _VSTD::distance(__first, __last);
    pair<value_type*, ptrdiff_t> __p(0, 0);
    unique_ptr<value_type, __return_temporary_buffer> __h;
    if (__len >= __alloc_limit)
    {
        __p = _VSTD::get_temporary_buffer<value_type>(__len);
        __h.reset(__p.first);
    }
    return _VSTD::__stable_partition<_Predicate&>(__first, __last, __pred, __len, __p, forward_iterator_tag());
}

template <class _Predicate, class _BidirectionalIterator, class _Distance, class _Pair>
_BidirectionalIterator
__stable_partition(_BidirectionalIterator __first, _BidirectionalIterator __last, _Predicate __pred,
                   _Distance __len, _Pair __p, bidirectional_iterator_tag __bit)
{
    // *__first is known to be false
    // *__last is known to be true
    // __len >= 2
    if (__len == 2)
    {
        swap(*__first, *__last);
        return __last;
    }
    if (__len == 3)
    {
        _BidirectionalIterator __m = __first;
        if (__pred(*++__m))
        {
            swap(*__first, *__m);
            swap(*__m, *__last);
            return __last;
        }
        swap(*__m, *__last);
        swap(*__first, *__m);
        return __m;
    }
    if (__len <= __p.second)
    {   // The buffer is big enough to use
        typedef typename iterator_traits<_BidirectionalIterator>::value_type value_type;
        __destruct_n __d(0);
        unique_ptr<value_type, __destruct_n&> __h(__p.first, __d);
        // Move the falses into the temporary buffer, and the trues to the front of the line
        // Update __first to always point to the end of the trues
        value_type* __t = __p.first;
        ::new ((void*)__t) value_type(_VSTD::move(*__first));
        __d.template __incr<value_type>();
        ++__t;
        _BidirectionalIterator __i = __first;
        while (++__i != __last)
        {
            if (__pred(*__i))
            {
                *__first = _VSTD::move(*__i);
                ++__first;
            }
            else
            {
                ::new ((void*)__t) value_type(_VSTD::move(*__i));
                __d.template __incr<value_type>();
                ++__t;
            }
        }
        // move *__last, known to be true
        *__first = _VSTD::move(*__i);
        __i = ++__first;
        // All trues now at start of range, all falses in buffer
        // Move falses back into range, but don't mess up __first which points to first false
        for (value_type* __t2 = __p.first; __t2 < __t; ++__t2, (void) ++__i)
            *__i = _VSTD::move(*__t2);
        // __h destructs moved-from values out of the temp buffer, but doesn't deallocate buffer
        return __first;
    }
    // Else not enough buffer, do in place
    // __len >= 4
    _BidirectionalIterator __m = __first;
    _Distance __len2 = __len / 2;  // __len2 >= 2
    _VSTD::advance(__m, __len2);
    // recurse on [__first, __m-1], except reduce __m-1 until *(__m-1) is true, *__first know to be false
    // F????????????????T
    // f       m        l
    _BidirectionalIterator __m1 = __m;
    _BidirectionalIterator __first_false = __first;
    _Distance __len_half = __len2;
    while (!__pred(*--__m1))
    {
        if (__m1 == __first)
            goto __first_half_done;
        --__len_half;
    }
    // F???TFFF?????????T
    // f   m1  m        l
    __first_false = _VSTD::__stable_partition<_Predicate&>(__first, __m1, __pred, __len_half, __p, __bit);
__first_half_done:
    // TTTFFFFF?????????T
    // f  ff   m        l
    // recurse on [__m, __last], except increase __m until *(__m) is false, *__last know to be true
    __m1 = __m;
    _BidirectionalIterator __second_false = __last;
    ++__second_false;
    __len_half = __len - __len2;
    while (__pred(*__m1))
    {
        if (++__m1 == __last)
            goto __second_half_done;
        --__len_half;
    }
    // TTTFFFFFTTTF?????T
    // f  ff   m  m1    l
    __second_false = _VSTD::__stable_partition<_Predicate&>(__m1, __last, __pred, __len_half, __p, __bit);
__second_half_done:
    // TTTFFFFFTTTTTFFFFF
    // f  ff   m    sf  l
    return _VSTD::rotate(__first_false, __m, __second_false);
    // TTTTTTTTFFFFFFFFFF
    //         |
}

template <class _Predicate, class _BidirectionalIterator>
_BidirectionalIterator
__stable_partition(_BidirectionalIterator __first, _BidirectionalIterator __last, _Predicate __pred,
                   bidirectional_iterator_tag)
{
    typedef typename iterator_traits<_BidirectionalIterator>::difference_type difference_type;
    typedef typename iterator_traits<_BidirectionalIterator>::value_type value_type;
    const difference_type __alloc_limit = 4;  // might want to make this a function of trivial assignment
    // Either prove all true and return __first or point to first false
    while (true)
    {
        if (__first == __last)
            return __first;
        if (!__pred(*__first))
            break;
        ++__first;
    }
    // __first points to first false, everything prior to __first is already set.
    // Either prove [__first, __last) is all false and return __first, or point __last to last true
    do
    {
        if (__first == --__last)
            return __first;
    } while (!__pred(*__last));
    // We now have a reduced range [__first, __last]
    // *__first is known to be false
    // *__last is known to be true
    // __len >= 2
    difference_type __len = _VSTD::distance(__first, __last) + 1;
    pair<value_type*, ptrdiff_t> __p(0, 0);
    unique_ptr<value_type, __return_temporary_buffer> __h;
    if (__len >= __alloc_limit)
    {
        __p = _VSTD::get_temporary_buffer<value_type>(__len);
        __h.reset(__p.first);
    }
    return _VSTD::__stable_partition<_Predicate&>(__first, __last, __pred, __len, __p, bidirectional_iterator_tag());
}

template <class _ForwardIterator, class _Predicate>
inline _LIBCPP_INLINE_VISIBILITY
_ForwardIterator
stable_partition(_ForwardIterator __first, _ForwardIterator __last, _Predicate __pred)
{
    return _VSTD::__stable_partition<_Predicate&>(__first, __last, __pred, typename iterator_traits<_ForwardIterator>::iterator_category());
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_STABLE_PARTITION_H
