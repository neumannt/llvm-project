// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CHRONO_DURATION_H
#define _LIBCPP___CHRONO_DURATION_H

#include <__config>
#include <limits>
#include <ratio>
#include <type_traits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#  pragma clang include_instead(<chrono>)
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

namespace chrono
{

template <class _Rep, class _Period = ratio<1> > class _LIBCPP_TEMPLATE_VIS duration;

template <class _Tp>
struct __is_duration : false_type {};

template <class _Rep, class _Period>
struct __is_duration<duration<_Rep, _Period> > : true_type  {};

template <class _Rep, class _Period>
struct __is_duration<const duration<_Rep, _Period> > : true_type  {};

template <class _Rep, class _Period>
struct __is_duration<volatile duration<_Rep, _Period> > : true_type  {};

template <class _Rep, class _Period>
struct __is_duration<const volatile duration<_Rep, _Period> > : true_type  {};

} // namespace chrono

template <class _Rep1, class _Period1, class _Rep2, class _Period2>
struct _LIBCPP_TEMPLATE_VIS common_type<chrono::duration<_Rep1, _Period1>,
                                         chrono::duration<_Rep2, _Period2> >
{
    typedef chrono::duration<typename common_type<_Rep1, _Rep2>::type,
                             typename __ratio_gcd<_Period1, _Period2>::type> type;
};

namespace chrono {

// duration_cast

template <class _FromDuration, class _ToDuration,
          class _Period = typename ratio_divide<typename _FromDuration::period, typename _ToDuration::period>::type,
          bool = _Period::num == 1,
          bool = _Period::den == 1>
struct __duration_cast;

template <class _FromDuration, class _ToDuration, class _Period>
struct __duration_cast<_FromDuration, _ToDuration, _Period, true, true>
{
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR
    _ToDuration operator()(const _FromDuration& __fd) const
    {
        return _ToDuration(static_cast<typename _ToDuration::rep>(__fd.count()));
    }
};

template <class _FromDuration, class _ToDuration, class _Period>
struct __duration_cast<_FromDuration, _ToDuration, _Period, true, false>
{
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR
    _ToDuration operator()(const _FromDuration& __fd) const
    {
        typedef typename common_type<typename _ToDuration::rep, typename _FromDuration::rep, intmax_t>::type _Ct;
        return _ToDuration(static_cast<typename _ToDuration::rep>(
                           static_cast<_Ct>(__fd.count()) / static_cast<_Ct>(_Period::den)));
    }
};

template <class _FromDuration, class _ToDuration, class _Period>
struct __duration_cast<_FromDuration, _ToDuration, _Period, false, true>
{
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR
    _ToDuration operator()(const _FromDuration& __fd) const
    {
        typedef typename common_type<typename _ToDuration::rep, typename _FromDuration::rep, intmax_t>::type _Ct;
        return _ToDuration(static_cast<typename _ToDuration::rep>(
                           static_cast<_Ct>(__fd.count()) * static_cast<_Ct>(_Period::num)));
    }
};

template <class _FromDuration, class _ToDuration, class _Period>
struct __duration_cast<_FromDuration, _ToDuration, _Period, false, false>
{
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR
    _ToDuration operator()(const _FromDuration& __fd) const
    {
        typedef typename common_type<typename _ToDuration::rep, typename _FromDuration::rep, intmax_t>::type _Ct;
        return _ToDuration(static_cast<typename _ToDuration::rep>(
                           static_cast<_Ct>(__fd.count()) * static_cast<_Ct>(_Period::num)
                                                          / static_cast<_Ct>(_Period::den)));
    }
};

template <class _ToDuration, class _Rep, class _Period>
inline _LIBCPP_INLINE_VISIBILITY
_LIBCPP_CONSTEXPR
typename enable_if
<
    __is_duration<_ToDuration>::value,
    _ToDuration
>::type
duration_cast(const duration<_Rep, _Period>& __fd)
{
    return __duration_cast<duration<_Rep, _Period>, _ToDuration>()(__fd);
}

template <class _Rep>
struct _LIBCPP_TEMPLATE_VIS treat_as_floating_point : is_floating_point<_Rep> {};

#if _LIBCPP_STD_VER > 14
template <class _Rep>
inline constexpr bool treat_as_floating_point_v = treat_as_floating_point<_Rep>::value;
#endif

template <class _Rep>
struct _LIBCPP_TEMPLATE_VIS duration_values
{
public:
    _LIBCPP_INLINE_VISIBILITY static _LIBCPP_CONSTEXPR _Rep zero() _NOEXCEPT {return _Rep(0);}
    _LIBCPP_INLINE_VISIBILITY static _LIBCPP_CONSTEXPR _Rep max()  _NOEXCEPT {return numeric_limits<_Rep>::max();}
    _LIBCPP_INLINE_VISIBILITY static _LIBCPP_CONSTEXPR _Rep min()  _NOEXCEPT {return numeric_limits<_Rep>::lowest();}
};

#if _LIBCPP_STD_VER > 14
template <class _ToDuration, class _Rep, class _Period>
inline _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR
typename enable_if
<
    __is_duration<_ToDuration>::value,
    _ToDuration
>::type
floor(const duration<_Rep, _Period>& __d)
{
    _ToDuration __t = duration_cast<_ToDuration>(__d);
    if (__t > __d)
        __t = __t - _ToDuration{1};
    return __t;
}

template <class _ToDuration, class _Rep, class _Period>
inline _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR
typename enable_if
<
    __is_duration<_ToDuration>::value,
    _ToDuration
>::type
ceil(const duration<_Rep, _Period>& __d)
{
    _ToDuration __t = duration_cast<_ToDuration>(__d);
    if (__t < __d)
        __t = __t + _ToDuration{1};
    return __t;
}

template <class _ToDuration, class _Rep, class _Period>
inline _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR
typename enable_if
<
    __is_duration<_ToDuration>::value,
    _ToDuration
>::type
round(const duration<_Rep, _Period>& __d)
{
    _ToDuration __lower = floor<_ToDuration>(__d);
    _ToDuration __upper = __lower + _ToDuration{1};
    auto __lowerDiff = __d - __lower;
    auto __upperDiff = __upper - __d;
    if (__lowerDiff < __upperDiff)
        return __lower;
    if (__lowerDiff > __upperDiff)
        return __upper;
    return __lower.count() & 1 ? __upper : __lower;
}
#endif

// duration

template <class _Rep, class _Period>
class _LIBCPP_TEMPLATE_VIS duration
{
    static_assert(!__is_duration<_Rep>::value, "A duration representation can not be a duration");
    static_assert(__is_ratio<_Period>::value, "Second template parameter of duration must be a std::ratio");
    static_assert(_Period::num > 0, "duration period must be positive");

    template <class _R1, class _R2>
    struct __no_overflow
    {
    private:
        static const intmax_t __gcd_n1_n2 = __static_gcd<_R1::num, _R2::num>::value;
        static const intmax_t __gcd_d1_d2 = __static_gcd<_R1::den, _R2::den>::value;
        static const intmax_t __n1 = _R1::num / __gcd_n1_n2;
        static const intmax_t __d1 = _R1::den / __gcd_d1_d2;
        static const intmax_t __n2 = _R2::num / __gcd_n1_n2;
        static const intmax_t __d2 = _R2::den / __gcd_d1_d2;
        static const intmax_t max = -((intmax_t(1) << (sizeof(intmax_t) * CHAR_BIT - 1)) + 1);

        template <intmax_t _Xp, intmax_t _Yp, bool __overflow>
        struct __mul    // __overflow == false
        {
            static const intmax_t value = _Xp * _Yp;
        };

        template <intmax_t _Xp, intmax_t _Yp>
        struct __mul<_Xp, _Yp, true>
        {
            static const intmax_t value = 1;
        };

    public:
        static const bool value = (__n1 <= max / __d2) && (__n2 <= max / __d1);
        typedef ratio<__mul<__n1, __d2, !value>::value,
                      __mul<__n2, __d1, !value>::value> type;
    };

public:
    typedef _Rep rep;
    typedef typename _Period::type period;
private:
    rep __rep_;
public:

    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR
#ifndef _LIBCPP_CXX03_LANG
        duration() = default;
#else
        duration() {}
#endif

    template <class _Rep2>
        _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR
        explicit duration(const _Rep2& __r,
            typename enable_if
            <
               is_convertible<const _Rep2&, rep>::value &&
               (treat_as_floating_point<rep>::value ||
               !treat_as_floating_point<_Rep2>::value)
            >::type* = nullptr)
                : __rep_(__r) {}

    // conversions
    template <class _Rep2, class _Period2>
        _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR
        duration(const duration<_Rep2, _Period2>& __d,
            typename enable_if
            <
                __no_overflow<_Period2, period>::value && (
                treat_as_floating_point<rep>::value ||
                (__no_overflow<_Period2, period>::type::den == 1 &&
                 !treat_as_floating_point<_Rep2>::value))
            >::type* = nullptr)
                : __rep_(chrono::duration_cast<duration>(__d).count()) {}

    // observer

    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR rep count() const {return __rep_;}

    // arithmetic

    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR typename common_type<duration>::type operator+() const {return typename common_type<duration>::type(*this);}
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR typename common_type<duration>::type operator-() const {return typename common_type<duration>::type(-__rep_);}
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14 duration& operator++()      {++__rep_; return *this;}
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14 duration  operator++(int)   {return duration(__rep_++);}
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14 duration& operator--()      {--__rep_; return *this;}
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14 duration  operator--(int)   {return duration(__rep_--);}

    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14 duration& operator+=(const duration& __d) {__rep_ += __d.count(); return *this;}
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14 duration& operator-=(const duration& __d) {__rep_ -= __d.count(); return *this;}

    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14 duration& operator*=(const rep& rhs) {__rep_ *= rhs; return *this;}
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14 duration& operator/=(const rep& rhs) {__rep_ /= rhs; return *this;}
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14 duration& operator%=(const rep& rhs) {__rep_ %= rhs; return *this;}
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14 duration& operator%=(const duration& rhs) {__rep_ %= rhs.count(); return *this;}

    // special values

    _LIBCPP_INLINE_VISIBILITY static _LIBCPP_CONSTEXPR duration zero() _NOEXCEPT {return duration(duration_values<rep>::zero());}
    _LIBCPP_INLINE_VISIBILITY static _LIBCPP_CONSTEXPR duration min()  _NOEXCEPT {return duration(duration_values<rep>::min());}
    _LIBCPP_INLINE_VISIBILITY static _LIBCPP_CONSTEXPR duration max()  _NOEXCEPT {return duration(duration_values<rep>::max());}
};

typedef duration<long long,         nano> nanoseconds;
typedef duration<long long,        micro> microseconds;
typedef duration<long long,        milli> milliseconds;
typedef duration<long long              > seconds;
typedef duration<     long, ratio<  60> > minutes;
typedef duration<     long, ratio<3600> > hours;
#if _LIBCPP_STD_VER > 17
typedef duration<     int, ratio_multiply<ratio<24>, hours::period>>         days;
typedef duration<     int, ratio_multiply<ratio<7>,   days::period>>         weeks;
typedef duration<     int, ratio_multiply<ratio<146097, 400>, days::period>> years;
typedef duration<     int, ratio_divide<years::period, ratio<12>>>           months;
#endif
// Duration ==

template <class _LhsDuration, class _RhsDuration>
struct __duration_eq
{
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR
    bool operator()(const _LhsDuration& __lhs, const _RhsDuration& __rhs) const
        {
            typedef typename common_type<_LhsDuration, _RhsDuration>::type _Ct;
            return _Ct(__lhs).count() == _Ct(__rhs).count();
        }
};

template <class _LhsDuration>
struct __duration_eq<_LhsDuration, _LhsDuration>
{
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR
    bool operator()(const _LhsDuration& __lhs, const _LhsDuration& __rhs) const
        {return __lhs.count() == __rhs.count();}
};

template <class _Rep1, class _Period1, class _Rep2, class _Period2>
inline _LIBCPP_INLINE_VISIBILITY
_LIBCPP_CONSTEXPR
bool
operator==(const duration<_Rep1, _Period1>& __lhs, const duration<_Rep2, _Period2>& __rhs)
{
    return __duration_eq<duration<_Rep1, _Period1>, duration<_Rep2, _Period2> >()(__lhs, __rhs);
}

// Duration !=

template <class _Rep1, class _Period1, class _Rep2, class _Period2>
inline _LIBCPP_INLINE_VISIBILITY
_LIBCPP_CONSTEXPR
bool
operator!=(const duration<_Rep1, _Period1>& __lhs, const duration<_Rep2, _Period2>& __rhs)
{
    return !(__lhs == __rhs);
}

// Duration <

template <class _LhsDuration, class _RhsDuration>
struct __duration_lt
{
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR
    bool operator()(const _LhsDuration& __lhs, const _RhsDuration& __rhs) const
        {
            typedef typename common_type<_LhsDuration, _RhsDuration>::type _Ct;
            return _Ct(__lhs).count() < _Ct(__rhs).count();
        }
};

template <class _LhsDuration>
struct __duration_lt<_LhsDuration, _LhsDuration>
{
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR
    bool operator()(const _LhsDuration& __lhs, const _LhsDuration& __rhs) const
        {return __lhs.count() < __rhs.count();}
};

template <class _Rep1, class _Period1, class _Rep2, class _Period2>
inline _LIBCPP_INLINE_VISIBILITY
_LIBCPP_CONSTEXPR
bool
operator< (const duration<_Rep1, _Period1>& __lhs, const duration<_Rep2, _Period2>& __rhs)
{
    return __duration_lt<duration<_Rep1, _Period1>, duration<_Rep2, _Period2> >()(__lhs, __rhs);
}

// Duration >

template <class _Rep1, class _Period1, class _Rep2, class _Period2>
inline _LIBCPP_INLINE_VISIBILITY
_LIBCPP_CONSTEXPR
bool
operator> (const duration<_Rep1, _Period1>& __lhs, const duration<_Rep2, _Period2>& __rhs)
{
    return __rhs < __lhs;
}

// Duration <=

template <class _Rep1, class _Period1, class _Rep2, class _Period2>
inline _LIBCPP_INLINE_VISIBILITY
_LIBCPP_CONSTEXPR
bool
operator<=(const duration<_Rep1, _Period1>& __lhs, const duration<_Rep2, _Period2>& __rhs)
{
    return !(__rhs < __lhs);
}

// Duration >=

template <class _Rep1, class _Period1, class _Rep2, class _Period2>
inline _LIBCPP_INLINE_VISIBILITY
_LIBCPP_CONSTEXPR
bool
operator>=(const duration<_Rep1, _Period1>& __lhs, const duration<_Rep2, _Period2>& __rhs)
{
    return !(__lhs < __rhs);
}

// Duration +

template <class _Rep1, class _Period1, class _Rep2, class _Period2>
inline _LIBCPP_INLINE_VISIBILITY
_LIBCPP_CONSTEXPR
typename common_type<duration<_Rep1, _Period1>, duration<_Rep2, _Period2> >::type
operator+(const duration<_Rep1, _Period1>& __lhs, const duration<_Rep2, _Period2>& __rhs)
{
    typedef typename common_type<duration<_Rep1, _Period1>, duration<_Rep2, _Period2> >::type _Cd;
    return _Cd(_Cd(__lhs).count() + _Cd(__rhs).count());
}

// Duration -

template <class _Rep1, class _Period1, class _Rep2, class _Period2>
inline _LIBCPP_INLINE_VISIBILITY
_LIBCPP_CONSTEXPR
typename common_type<duration<_Rep1, _Period1>, duration<_Rep2, _Period2> >::type
operator-(const duration<_Rep1, _Period1>& __lhs, const duration<_Rep2, _Period2>& __rhs)
{
    typedef typename common_type<duration<_Rep1, _Period1>, duration<_Rep2, _Period2> >::type _Cd;
    return _Cd(_Cd(__lhs).count() - _Cd(__rhs).count());
}

// Duration *

template <class _Rep1, class _Period, class _Rep2>
inline _LIBCPP_INLINE_VISIBILITY
_LIBCPP_CONSTEXPR
typename enable_if
<
    is_convertible<_Rep2, typename common_type<_Rep1, _Rep2>::type>::value,
    duration<typename common_type<_Rep1, _Rep2>::type, _Period>
>::type
operator*(const duration<_Rep1, _Period>& __d, const _Rep2& __s)
{
    typedef typename common_type<_Rep1, _Rep2>::type _Cr;
    typedef duration<_Cr, _Period> _Cd;
    return _Cd(_Cd(__d).count() * static_cast<_Cr>(__s));
}

template <class _Rep1, class _Period, class _Rep2>
inline _LIBCPP_INLINE_VISIBILITY
_LIBCPP_CONSTEXPR
typename enable_if
<
    is_convertible<_Rep1, typename common_type<_Rep1, _Rep2>::type>::value,
    duration<typename common_type<_Rep1, _Rep2>::type, _Period>
>::type
operator*(const _Rep1& __s, const duration<_Rep2, _Period>& __d)
{
    return __d * __s;
}

// Duration /

template <class _Rep1, class _Period, class _Rep2>
inline _LIBCPP_INLINE_VISIBILITY
_LIBCPP_CONSTEXPR
typename enable_if
<
    !__is_duration<_Rep2>::value &&
      is_convertible<_Rep2, typename common_type<_Rep1, _Rep2>::type>::value,
    duration<typename common_type<_Rep1, _Rep2>::type, _Period>
>::type
operator/(const duration<_Rep1, _Period>& __d, const _Rep2& __s)
{
    typedef typename common_type<_Rep1, _Rep2>::type _Cr;
    typedef duration<_Cr, _Period> _Cd;
    return _Cd(_Cd(__d).count() / static_cast<_Cr>(__s));
}

template <class _Rep1, class _Period1, class _Rep2, class _Period2>
inline _LIBCPP_INLINE_VISIBILITY
_LIBCPP_CONSTEXPR
typename common_type<_Rep1, _Rep2>::type
operator/(const duration<_Rep1, _Period1>& __lhs, const duration<_Rep2, _Period2>& __rhs)
{
    typedef typename common_type<duration<_Rep1, _Period1>, duration<_Rep2, _Period2> >::type _Ct;
    return _Ct(__lhs).count() / _Ct(__rhs).count();
}

// Duration %

template <class _Rep1, class _Period, class _Rep2>
inline _LIBCPP_INLINE_VISIBILITY
_LIBCPP_CONSTEXPR
typename enable_if
<
    !__is_duration<_Rep2>::value &&
      is_convertible<_Rep2, typename common_type<_Rep1, _Rep2>::type>::value,
    duration<typename common_type<_Rep1, _Rep2>::type, _Period>
>::type
operator%(const duration<_Rep1, _Period>& __d, const _Rep2& __s)
{
    typedef typename common_type<_Rep1, _Rep2>::type _Cr;
    typedef duration<_Cr, _Period> _Cd;
    return _Cd(_Cd(__d).count() % static_cast<_Cr>(__s));
}

template <class _Rep1, class _Period1, class _Rep2, class _Period2>
inline _LIBCPP_INLINE_VISIBILITY
_LIBCPP_CONSTEXPR
typename common_type<duration<_Rep1, _Period1>, duration<_Rep2, _Period2> >::type
operator%(const duration<_Rep1, _Period1>& __lhs, const duration<_Rep2, _Period2>& __rhs)
{
    typedef typename common_type<_Rep1, _Rep2>::type _Cr;
    typedef typename common_type<duration<_Rep1, _Period1>, duration<_Rep2, _Period2> >::type _Cd;
    return _Cd(static_cast<_Cr>(_Cd(__lhs).count()) % static_cast<_Cr>(_Cd(__rhs).count()));
}

} // namespace chrono

#if _LIBCPP_STD_VER > 11
// Suffixes for duration literals [time.duration.literals]
inline namespace literals
{
  inline namespace chrono_literals
  {

    constexpr chrono::hours operator""h(unsigned long long __h)
    {
        return chrono::hours(static_cast<chrono::hours::rep>(__h));
    }

    constexpr chrono::duration<long double, ratio<3600,1>> operator""h(long double __h)
    {
        return chrono::duration<long double, ratio<3600,1>>(__h);
    }


    constexpr chrono::minutes operator""min(unsigned long long __m)
    {
        return chrono::minutes(static_cast<chrono::minutes::rep>(__m));
    }

    constexpr chrono::duration<long double, ratio<60,1>> operator""min(long double __m)
    {
        return chrono::duration<long double, ratio<60,1>> (__m);
    }


    constexpr chrono::seconds operator""s(unsigned long long __s)
    {
        return chrono::seconds(static_cast<chrono::seconds::rep>(__s));
    }

    constexpr chrono::duration<long double> operator""s(long double __s)
    {
        return chrono::duration<long double> (__s);
    }


    constexpr chrono::milliseconds operator""ms(unsigned long long __ms)
    {
        return chrono::milliseconds(static_cast<chrono::milliseconds::rep>(__ms));
    }

    constexpr chrono::duration<long double, milli> operator""ms(long double __ms)
    {
        return chrono::duration<long double, milli>(__ms);
    }


    constexpr chrono::microseconds operator""us(unsigned long long __us)
    {
        return chrono::microseconds(static_cast<chrono::microseconds::rep>(__us));
    }

    constexpr chrono::duration<long double, micro> operator""us(long double __us)
    {
        return chrono::duration<long double, micro> (__us);
    }


    constexpr chrono::nanoseconds operator""ns(unsigned long long __ns)
    {
        return chrono::nanoseconds(static_cast<chrono::nanoseconds::rep>(__ns));
    }

    constexpr chrono::duration<long double, nano> operator""ns(long double __ns)
    {
        return chrono::duration<long double, nano> (__ns);
    }

} // namespace chrono_literals
} // namespace literals

namespace chrono { // hoist the literals into namespace std::chrono
   using namespace literals::chrono_literals;
} // namespace chrono

#endif // _LIBCPP_STD_VER > 11

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___CHRONO_DURATION_H
