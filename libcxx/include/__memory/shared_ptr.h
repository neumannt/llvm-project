// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_SHARED_PTR_H
#define _LIBCPP___MEMORY_SHARED_PTR_H

#include <__availability>
#include <__config>
#include <__functional/binary_function.h>
#include <__functional/operations.h>
#include <__functional/reference_wrapper.h>
#include <__memory/addressof.h>
#include <__memory/allocation_guard.h>
#include <__memory/allocator.h>
#include <__memory/allocator_traits.h>
#include <__memory/compressed_pair.h>
#include <__memory/pointer_traits.h>
#include <__memory/unique_ptr.h>
#include <__utility/forward.h>
#include <__utility/move.h>
#include <cstddef>
#include <cstdlib> // abort
#include <iosfwd>
#include <stdexcept>
#include <type_traits>
#include <typeinfo>
#if !defined(_LIBCPP_HAS_NO_ATOMIC_HEADER)
#  include <atomic>
#endif

#if _LIBCPP_STD_VER <= 14 || defined(_LIBCPP_ENABLE_CXX17_REMOVED_AUTO_PTR)
#   include <__memory/auto_ptr.h>
#endif

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#  pragma clang include_instead(<memory>)
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Alloc>
class __allocator_destructor
{
    typedef _LIBCPP_NODEBUG allocator_traits<_Alloc> __alloc_traits;
public:
    typedef _LIBCPP_NODEBUG typename __alloc_traits::pointer pointer;
    typedef _LIBCPP_NODEBUG typename __alloc_traits::size_type size_type;
private:
    _Alloc& __alloc_;
    size_type __s_;
public:
    _LIBCPP_INLINE_VISIBILITY __allocator_destructor(_Alloc& __a, size_type __s)
             _NOEXCEPT
        : __alloc_(__a), __s_(__s) {}
    _LIBCPP_INLINE_VISIBILITY
    void operator()(pointer __p) _NOEXCEPT
        {__alloc_traits::deallocate(__alloc_, __p, __s_);}
};

// NOTE: Relaxed and acq/rel atomics (for increment and decrement respectively)
// should be sufficient for thread safety.
// See https://llvm.org/PR22803
#if defined(__clang__) && __has_builtin(__atomic_add_fetch)          \
                       && defined(__ATOMIC_RELAXED)                  \
                       && defined(__ATOMIC_ACQ_REL)
#   define _LIBCPP_HAS_BUILTIN_ATOMIC_SUPPORT
#elif defined(_LIBCPP_COMPILER_GCC)
#   define _LIBCPP_HAS_BUILTIN_ATOMIC_SUPPORT
#endif

template <class _ValueType>
inline _LIBCPP_INLINE_VISIBILITY
_ValueType __libcpp_relaxed_load(_ValueType const* __value) {
#if !defined(_LIBCPP_HAS_NO_THREADS) && \
    defined(__ATOMIC_RELAXED) &&        \
    (__has_builtin(__atomic_load_n) || defined(_LIBCPP_COMPILER_GCC))
    return __atomic_load_n(__value, __ATOMIC_RELAXED);
#else
    return *__value;
#endif
}

template <class _ValueType>
inline _LIBCPP_INLINE_VISIBILITY
_ValueType __libcpp_acquire_load(_ValueType const* __value) {
#if !defined(_LIBCPP_HAS_NO_THREADS) && \
    defined(__ATOMIC_ACQUIRE) &&        \
    (__has_builtin(__atomic_load_n) || defined(_LIBCPP_COMPILER_GCC))
    return __atomic_load_n(__value, __ATOMIC_ACQUIRE);
#else
    return *__value;
#endif
}

template <class _Tp>
inline _LIBCPP_INLINE_VISIBILITY _Tp
__libcpp_atomic_refcount_increment(_Tp& __t) _NOEXCEPT
{
#if defined(_LIBCPP_HAS_BUILTIN_ATOMIC_SUPPORT) && !defined(_LIBCPP_HAS_NO_THREADS)
    return __atomic_add_fetch(&__t, 1, __ATOMIC_RELAXED);
#else
    return __t += 1;
#endif
}

template <class _Tp>
inline _LIBCPP_INLINE_VISIBILITY _Tp
__libcpp_atomic_refcount_decrement(_Tp& __t) _NOEXCEPT
{
#if defined(_LIBCPP_HAS_BUILTIN_ATOMIC_SUPPORT) && !defined(_LIBCPP_HAS_NO_THREADS)
    return __atomic_add_fetch(&__t, -1, __ATOMIC_ACQ_REL);
#else
    return __t -= 1;
#endif
}

class _LIBCPP_EXCEPTION_ABI bad_weak_ptr
    : public std::exception
{
public:
    bad_weak_ptr() _NOEXCEPT = default;
    bad_weak_ptr(const bad_weak_ptr&) _NOEXCEPT = default;
    virtual ~bad_weak_ptr() _NOEXCEPT;
    virtual const char* what() const  _NOEXCEPT;
};

_LIBCPP_NORETURN inline _LIBCPP_INLINE_VISIBILITY
void __throw_bad_weak_ptr()
{
#ifndef _LIBCPP_NO_EXCEPTIONS
    throw bad_weak_ptr();
#else
    _VSTD::abort();
#endif
}

template<class _Tp> class _LIBCPP_TEMPLATE_VIS weak_ptr;

class _LIBCPP_TYPE_VIS __shared_count
{
    __shared_count(const __shared_count&);
    __shared_count& operator=(const __shared_count&);

protected:
    long __shared_owners_;
    virtual ~__shared_count();
private:
    virtual void __on_zero_shared() _NOEXCEPT = 0;

public:
    _LIBCPP_INLINE_VISIBILITY
    explicit __shared_count(long __refs = 0) _NOEXCEPT
        : __shared_owners_(__refs) {}

#if defined(_LIBCPP_BUILDING_LIBRARY) && \
    defined(_LIBCPP_DEPRECATED_ABI_LEGACY_LIBRARY_DEFINITIONS_FOR_INLINE_FUNCTIONS)
    void __add_shared() _NOEXCEPT;
    bool __release_shared() _NOEXCEPT;
#else
    _LIBCPP_INLINE_VISIBILITY
    void __add_shared() _NOEXCEPT {
      __libcpp_atomic_refcount_increment(__shared_owners_);
    }
    _LIBCPP_INLINE_VISIBILITY
    bool __release_shared() _NOEXCEPT {
      if (__libcpp_atomic_refcount_decrement(__shared_owners_) == -1) {
        __on_zero_shared();
        return true;
      }
      return false;
    }
#endif
    _LIBCPP_INLINE_VISIBILITY
    long use_count() const _NOEXCEPT {
        return __libcpp_relaxed_load(&__shared_owners_) + 1;
    }
};

class _LIBCPP_TYPE_VIS __shared_weak_count
    : private __shared_count
{
    long __shared_weak_owners_;

public:
    _LIBCPP_INLINE_VISIBILITY
    explicit __shared_weak_count(long __refs = 0) _NOEXCEPT
        : __shared_count(__refs),
          __shared_weak_owners_(__refs) {}
protected:
    virtual ~__shared_weak_count();

public:
#if defined(_LIBCPP_BUILDING_LIBRARY) && \
    defined(_LIBCPP_DEPRECATED_ABI_LEGACY_LIBRARY_DEFINITIONS_FOR_INLINE_FUNCTIONS)
    void __add_shared() _NOEXCEPT;
    void __add_weak() _NOEXCEPT;
    void __release_shared() _NOEXCEPT;
#else
    _LIBCPP_INLINE_VISIBILITY
    void __add_shared() _NOEXCEPT {
      __shared_count::__add_shared();
    }
    _LIBCPP_INLINE_VISIBILITY
    void __add_weak() _NOEXCEPT {
      __libcpp_atomic_refcount_increment(__shared_weak_owners_);
    }
    _LIBCPP_INLINE_VISIBILITY
    void __release_shared() _NOEXCEPT {
      if (__shared_count::__release_shared())
        __release_weak();
    }
#endif
    void __release_weak() _NOEXCEPT;
    _LIBCPP_INLINE_VISIBILITY
    long use_count() const _NOEXCEPT {return __shared_count::use_count();}
    __shared_weak_count* lock() _NOEXCEPT;

    virtual const void* __get_deleter(const type_info&) const _NOEXCEPT;
private:
    virtual void __on_zero_shared_weak() _NOEXCEPT = 0;
};

template <class _Tp, class _Dp, class _Alloc>
class __shared_ptr_pointer
    : public __shared_weak_count
{
    __compressed_pair<__compressed_pair<_Tp, _Dp>, _Alloc> __data_;
public:
    _LIBCPP_INLINE_VISIBILITY
    __shared_ptr_pointer(_Tp __p, _Dp __d, _Alloc __a)
        :  __data_(__compressed_pair<_Tp, _Dp>(__p, _VSTD::move(__d)), _VSTD::move(__a)) {}

#ifndef _LIBCPP_NO_RTTI
    virtual const void* __get_deleter(const type_info&) const _NOEXCEPT;
#endif

private:
    virtual void __on_zero_shared() _NOEXCEPT;
    virtual void __on_zero_shared_weak() _NOEXCEPT;
};

#ifndef _LIBCPP_NO_RTTI

template <class _Tp, class _Dp, class _Alloc>
const void*
__shared_ptr_pointer<_Tp, _Dp, _Alloc>::__get_deleter(const type_info& __t) const _NOEXCEPT
{
    return __t == typeid(_Dp) ? _VSTD::addressof(__data_.first().second()) : nullptr;
}

#endif // _LIBCPP_NO_RTTI

template <class _Tp, class _Dp, class _Alloc>
void
__shared_ptr_pointer<_Tp, _Dp, _Alloc>::__on_zero_shared() _NOEXCEPT
{
    __data_.first().second()(__data_.first().first());
    __data_.first().second().~_Dp();
}

template <class _Tp, class _Dp, class _Alloc>
void
__shared_ptr_pointer<_Tp, _Dp, _Alloc>::__on_zero_shared_weak() _NOEXCEPT
{
    typedef typename __allocator_traits_rebind<_Alloc, __shared_ptr_pointer>::type _Al;
    typedef allocator_traits<_Al> _ATraits;
    typedef pointer_traits<typename _ATraits::pointer> _PTraits;

    _Al __a(__data_.second());
    __data_.second().~_Alloc();
    __a.deallocate(_PTraits::pointer_to(*this), 1);
}

template <class _Tp, class _Alloc>
struct __shared_ptr_emplace
    : __shared_weak_count
{
    template<class ..._Args>
    _LIBCPP_HIDE_FROM_ABI
    explicit __shared_ptr_emplace(_Alloc __a, _Args&& ...__args)
        : __storage_(_VSTD::move(__a))
    {
#if _LIBCPP_STD_VER > 17
        using _TpAlloc = typename __allocator_traits_rebind<_Alloc, _Tp>::type;
        _TpAlloc __tmp(*__get_alloc());
        allocator_traits<_TpAlloc>::construct(__tmp, __get_elem(), _VSTD::forward<_Args>(__args)...);
#else
        ::new ((void*)__get_elem()) _Tp(_VSTD::forward<_Args>(__args)...);
#endif
    }

    _LIBCPP_HIDE_FROM_ABI
    _Alloc* __get_alloc() _NOEXCEPT { return __storage_.__get_alloc(); }

    _LIBCPP_HIDE_FROM_ABI
    _Tp* __get_elem() _NOEXCEPT { return __storage_.__get_elem(); }

private:
    virtual void __on_zero_shared() _NOEXCEPT {
#if _LIBCPP_STD_VER > 17
        using _TpAlloc = typename __allocator_traits_rebind<_Alloc, _Tp>::type;
        _TpAlloc __tmp(*__get_alloc());
        allocator_traits<_TpAlloc>::destroy(__tmp, __get_elem());
#else
        __get_elem()->~_Tp();
#endif
    }

    virtual void __on_zero_shared_weak() _NOEXCEPT {
        using _ControlBlockAlloc = typename __allocator_traits_rebind<_Alloc, __shared_ptr_emplace>::type;
        using _ControlBlockPointer = typename allocator_traits<_ControlBlockAlloc>::pointer;
        _ControlBlockAlloc __tmp(*__get_alloc());
        __storage_.~_Storage();
        allocator_traits<_ControlBlockAlloc>::deallocate(__tmp,
            pointer_traits<_ControlBlockPointer>::pointer_to(*this), 1);
    }

    // This class implements the control block for non-array shared pointers created
    // through `std::allocate_shared` and `std::make_shared`.
    //
    // In previous versions of the library, we used a compressed pair to store
    // both the _Alloc and the _Tp. This implies using EBO, which is incompatible
    // with Allocator construction for _Tp. To allow implementing P0674 in C++20,
    // we now use a properly aligned char buffer while making sure that we maintain
    // the same layout that we had when we used a compressed pair.
    using _CompressedPair = __compressed_pair<_Alloc, _Tp>;
    struct _ALIGNAS_TYPE(_CompressedPair) _Storage {
        char __blob_[sizeof(_CompressedPair)];

        _LIBCPP_HIDE_FROM_ABI explicit _Storage(_Alloc&& __a) {
            ::new ((void*)__get_alloc()) _Alloc(_VSTD::move(__a));
        }
        _LIBCPP_HIDE_FROM_ABI ~_Storage() {
            __get_alloc()->~_Alloc();
        }
        _Alloc* __get_alloc() _NOEXCEPT {
            _CompressedPair *__as_pair = reinterpret_cast<_CompressedPair*>(__blob_);
            typename _CompressedPair::_Base1* __first = _CompressedPair::__get_first_base(__as_pair);
            _Alloc *__alloc = reinterpret_cast<_Alloc*>(__first);
            return __alloc;
        }
        _LIBCPP_NO_CFI _Tp* __get_elem() _NOEXCEPT {
            _CompressedPair *__as_pair = reinterpret_cast<_CompressedPair*>(__blob_);
            typename _CompressedPair::_Base2* __second = _CompressedPair::__get_second_base(__as_pair);
            _Tp *__elem = reinterpret_cast<_Tp*>(__second);
            return __elem;
        }
    };

    static_assert(_LIBCPP_ALIGNOF(_Storage) == _LIBCPP_ALIGNOF(_CompressedPair), "");
    static_assert(sizeof(_Storage) == sizeof(_CompressedPair), "");
    _Storage __storage_;
};

struct __shared_ptr_dummy_rebind_allocator_type;
template <>
class _LIBCPP_TEMPLATE_VIS allocator<__shared_ptr_dummy_rebind_allocator_type>
{
public:
    template <class _Other>
    struct rebind
    {
        typedef allocator<_Other> other;
    };
};

template<class _Tp> class _LIBCPP_TEMPLATE_VIS enable_shared_from_this;

template<class _Tp, class _Up>
struct __compatible_with
#if _LIBCPP_STD_VER > 14
    : is_convertible<remove_extent_t<_Tp>*, remove_extent_t<_Up>*> {};
#else
    : is_convertible<_Tp*, _Up*> {};
#endif // _LIBCPP_STD_VER > 14

template <class _Ptr, class = void>
struct __is_deletable : false_type { };
template <class _Ptr>
struct __is_deletable<_Ptr, decltype(delete declval<_Ptr>())> : true_type { };

template <class _Ptr, class = void>
struct __is_array_deletable : false_type { };
template <class _Ptr>
struct __is_array_deletable<_Ptr, decltype(delete[] declval<_Ptr>())> : true_type { };

template <class _Dp, class _Pt,
    class = decltype(declval<_Dp>()(declval<_Pt>()))>
static true_type __well_formed_deleter_test(int);

template <class, class>
static false_type __well_formed_deleter_test(...);

template <class _Dp, class _Pt>
struct __well_formed_deleter : decltype(__well_formed_deleter_test<_Dp, _Pt>(0)) {};

template<class _Dp, class _Tp, class _Yp>
struct __shared_ptr_deleter_ctor_reqs
{
    static const bool value = __compatible_with<_Tp, _Yp>::value &&
                              is_move_constructible<_Dp>::value &&
                              __well_formed_deleter<_Dp, _Tp*>::value;
};

#if defined(_LIBCPP_ABI_ENABLE_SHARED_PTR_TRIVIAL_ABI)
#  define _LIBCPP_SHARED_PTR_TRIVIAL_ABI __attribute__((trivial_abi))
#else
#  define _LIBCPP_SHARED_PTR_TRIVIAL_ABI
#endif

template<class _Tp>
class _LIBCPP_SHARED_PTR_TRIVIAL_ABI _LIBCPP_TEMPLATE_VIS shared_ptr
{
public:
#if _LIBCPP_STD_VER > 14
    typedef weak_ptr<_Tp> weak_type;
    typedef remove_extent_t<_Tp> element_type;
#else
    typedef _Tp element_type;
#endif

private:
    element_type*      __ptr_;
    __shared_weak_count* __cntrl_;

public:
    _LIBCPP_HIDE_FROM_ABI
    _LIBCPP_CONSTEXPR shared_ptr() _NOEXCEPT
        : __ptr_(nullptr),
          __cntrl_(nullptr)
    { }

    _LIBCPP_HIDE_FROM_ABI
    _LIBCPP_CONSTEXPR shared_ptr(nullptr_t) _NOEXCEPT
        : __ptr_(nullptr),
          __cntrl_(nullptr)
    { }

    template<class _Yp, class = __enable_if_t<
        _And<
            __compatible_with<_Yp, _Tp>
            // In C++03 we get errors when trying to do SFINAE with the
            // delete operator, so we always pretend that it's deletable.
            // The same happens on GCC.
#if !defined(_LIBCPP_CXX03_LANG) && !defined(_LIBCPP_COMPILER_GCC)
            , _If<is_array<_Tp>::value, __is_array_deletable<_Yp*>, __is_deletable<_Yp*> >
#endif
        >::value
    > >
    explicit shared_ptr(_Yp* __p) : __ptr_(__p) {
        unique_ptr<_Yp> __hold(__p);
        typedef typename __shared_ptr_default_allocator<_Yp>::type _AllocT;
        typedef __shared_ptr_pointer<_Yp*, __shared_ptr_default_delete<_Tp, _Yp>, _AllocT > _CntrlBlk;
        __cntrl_ = new _CntrlBlk(__p, __shared_ptr_default_delete<_Tp, _Yp>(), _AllocT());
        __hold.release();
        __enable_weak_this(__p, __p);
    }

    template<class _Yp, class _Dp, class = __enable_if_t<__shared_ptr_deleter_ctor_reqs<_Dp, _Yp, element_type>::value> >
    _LIBCPP_HIDE_FROM_ABI
    shared_ptr(_Yp* __p, _Dp __d)
        : __ptr_(__p)
    {
#ifndef _LIBCPP_NO_EXCEPTIONS
        try
        {
#endif // _LIBCPP_NO_EXCEPTIONS
            typedef typename __shared_ptr_default_allocator<_Yp>::type _AllocT;
            typedef __shared_ptr_pointer<_Yp*, _Dp, _AllocT > _CntrlBlk;
#ifndef _LIBCPP_CXX03_LANG
            __cntrl_ = new _CntrlBlk(__p, _VSTD::move(__d), _AllocT());
#else
            __cntrl_ = new _CntrlBlk(__p, __d, _AllocT());
#endif // not _LIBCPP_CXX03_LANG
            __enable_weak_this(__p, __p);
#ifndef _LIBCPP_NO_EXCEPTIONS
        }
        catch (...)
        {
            __d(__p);
            throw;
        }
#endif // _LIBCPP_NO_EXCEPTIONS
    }

    template<class _Yp, class _Dp, class _Alloc, class = __enable_if_t<__shared_ptr_deleter_ctor_reqs<_Dp, _Yp, element_type>::value> >
    _LIBCPP_HIDE_FROM_ABI
    shared_ptr(_Yp* __p, _Dp __d, _Alloc __a)
        : __ptr_(__p)
    {
#ifndef _LIBCPP_NO_EXCEPTIONS
        try
        {
#endif // _LIBCPP_NO_EXCEPTIONS
            typedef __shared_ptr_pointer<_Yp*, _Dp, _Alloc> _CntrlBlk;
            typedef typename __allocator_traits_rebind<_Alloc, _CntrlBlk>::type _A2;
            typedef __allocator_destructor<_A2> _D2;
            _A2 __a2(__a);
            unique_ptr<_CntrlBlk, _D2> __hold2(__a2.allocate(1), _D2(__a2, 1));
            ::new ((void*)_VSTD::addressof(*__hold2.get()))
#ifndef _LIBCPP_CXX03_LANG
                _CntrlBlk(__p, _VSTD::move(__d), __a);
#else
                _CntrlBlk(__p, __d, __a);
#endif // not _LIBCPP_CXX03_LANG
            __cntrl_ = _VSTD::addressof(*__hold2.release());
            __enable_weak_this(__p, __p);
#ifndef _LIBCPP_NO_EXCEPTIONS
        }
        catch (...)
        {
            __d(__p);
            throw;
        }
#endif // _LIBCPP_NO_EXCEPTIONS
    }

    template<class _Dp>
    _LIBCPP_HIDE_FROM_ABI
    shared_ptr(nullptr_t __p, _Dp __d)
        : __ptr_(nullptr)
    {
#ifndef _LIBCPP_NO_EXCEPTIONS
        try
        {
#endif // _LIBCPP_NO_EXCEPTIONS
            typedef typename __shared_ptr_default_allocator<_Tp>::type _AllocT;
            typedef __shared_ptr_pointer<nullptr_t, _Dp, _AllocT > _CntrlBlk;
#ifndef _LIBCPP_CXX03_LANG
            __cntrl_ = new _CntrlBlk(__p, _VSTD::move(__d), _AllocT());
#else
            __cntrl_ = new _CntrlBlk(__p, __d, _AllocT());
#endif // not _LIBCPP_CXX03_LANG
#ifndef _LIBCPP_NO_EXCEPTIONS
        }
        catch (...)
        {
            __d(__p);
            throw;
        }
#endif // _LIBCPP_NO_EXCEPTIONS
    }

    template<class _Dp, class _Alloc>
    _LIBCPP_HIDE_FROM_ABI
    shared_ptr(nullptr_t __p, _Dp __d, _Alloc __a)
        : __ptr_(nullptr)
    {
#ifndef _LIBCPP_NO_EXCEPTIONS
        try
        {
#endif // _LIBCPP_NO_EXCEPTIONS
            typedef __shared_ptr_pointer<nullptr_t, _Dp, _Alloc> _CntrlBlk;
            typedef typename __allocator_traits_rebind<_Alloc, _CntrlBlk>::type _A2;
            typedef __allocator_destructor<_A2> _D2;
            _A2 __a2(__a);
            unique_ptr<_CntrlBlk, _D2> __hold2(__a2.allocate(1), _D2(__a2, 1));
            ::new ((void*)_VSTD::addressof(*__hold2.get()))
#ifndef _LIBCPP_CXX03_LANG
                _CntrlBlk(__p, _VSTD::move(__d), __a);
#else
                _CntrlBlk(__p, __d, __a);
#endif // not _LIBCPP_CXX03_LANG
            __cntrl_ = _VSTD::addressof(*__hold2.release());
#ifndef _LIBCPP_NO_EXCEPTIONS
        }
        catch (...)
        {
            __d(__p);
            throw;
        }
#endif // _LIBCPP_NO_EXCEPTIONS
    }

    template<class _Yp>
    _LIBCPP_HIDE_FROM_ABI
    shared_ptr(const shared_ptr<_Yp>& __r, element_type *__p) _NOEXCEPT
        : __ptr_(__p),
          __cntrl_(__r.__cntrl_)
    {
        if (__cntrl_)
            __cntrl_->__add_shared();
    }

    _LIBCPP_HIDE_FROM_ABI
    shared_ptr(const shared_ptr& __r) _NOEXCEPT
        : __ptr_(__r.__ptr_),
          __cntrl_(__r.__cntrl_)
    {
        if (__cntrl_)
            __cntrl_->__add_shared();
    }

    template<class _Yp, class = __enable_if_t<__compatible_with<_Yp, _Tp>::value> >
    _LIBCPP_HIDE_FROM_ABI
    shared_ptr(const shared_ptr<_Yp>& __r) _NOEXCEPT
        : __ptr_(__r.__ptr_),
          __cntrl_(__r.__cntrl_)
    {
        if (__cntrl_)
            __cntrl_->__add_shared();
    }

    _LIBCPP_HIDE_FROM_ABI
    shared_ptr(shared_ptr&& __r) _NOEXCEPT
        : __ptr_(__r.__ptr_),
          __cntrl_(__r.__cntrl_)
    {
        __r.__ptr_ = nullptr;
        __r.__cntrl_ = nullptr;
    }

    template<class _Yp, class = __enable_if_t<__compatible_with<_Yp, _Tp>::value> >
    _LIBCPP_HIDE_FROM_ABI
    shared_ptr(shared_ptr<_Yp>&& __r) _NOEXCEPT
        : __ptr_(__r.__ptr_),
          __cntrl_(__r.__cntrl_)
    {
        __r.__ptr_ = nullptr;
        __r.__cntrl_ = nullptr;
    }

    template<class _Yp, class = __enable_if_t<__compatible_with<_Yp, _Tp>::value> >
    _LIBCPP_HIDE_FROM_ABI
    explicit shared_ptr(const weak_ptr<_Yp>& __r)
        : __ptr_(__r.__ptr_),
          __cntrl_(__r.__cntrl_ ? __r.__cntrl_->lock() : __r.__cntrl_)
    {
        if (__cntrl_ == nullptr)
            __throw_bad_weak_ptr();
    }

#if _LIBCPP_STD_VER <= 14 || defined(_LIBCPP_ENABLE_CXX17_REMOVED_AUTO_PTR)
    template<class _Yp, class = __enable_if_t<is_convertible<_Yp*, element_type*>::value> >
    _LIBCPP_HIDE_FROM_ABI
    shared_ptr(auto_ptr<_Yp>&& __r)
        : __ptr_(__r.get())
    {
        typedef __shared_ptr_pointer<_Yp*, default_delete<_Yp>, allocator<_Yp> > _CntrlBlk;
        __cntrl_ = new _CntrlBlk(__r.get(), default_delete<_Yp>(), allocator<_Yp>());
        __enable_weak_this(__r.get(), __r.get());
        __r.release();
    }
#endif

    template <class _Yp, class _Dp, class = __enable_if_t<
        !is_lvalue_reference<_Dp>::value &&
         is_convertible<typename unique_ptr<_Yp, _Dp>::pointer, element_type*>::value
    > >
    _LIBCPP_HIDE_FROM_ABI
    shared_ptr(unique_ptr<_Yp, _Dp>&& __r)
        : __ptr_(__r.get())
    {
#if _LIBCPP_STD_VER > 11
        if (__ptr_ == nullptr)
            __cntrl_ = nullptr;
        else
#endif
        {
            typedef typename __shared_ptr_default_allocator<_Yp>::type _AllocT;
            typedef __shared_ptr_pointer<typename unique_ptr<_Yp, _Dp>::pointer, _Dp, _AllocT > _CntrlBlk;
            __cntrl_ = new _CntrlBlk(__r.get(), __r.get_deleter(), _AllocT());
            __enable_weak_this(__r.get(), __r.get());
        }
        __r.release();
    }

    template <class _Yp, class _Dp, class = void, class = __enable_if_t<
        is_lvalue_reference<_Dp>::value &&
        is_convertible<typename unique_ptr<_Yp, _Dp>::pointer, element_type*>::value
    > >
    _LIBCPP_HIDE_FROM_ABI
    shared_ptr(unique_ptr<_Yp, _Dp>&& __r)
        : __ptr_(__r.get())
    {
#if _LIBCPP_STD_VER > 11
        if (__ptr_ == nullptr)
            __cntrl_ = nullptr;
        else
#endif
        {
            typedef typename __shared_ptr_default_allocator<_Yp>::type _AllocT;
            typedef __shared_ptr_pointer<typename unique_ptr<_Yp, _Dp>::pointer,
                                        reference_wrapper<typename remove_reference<_Dp>::type>,
                                        _AllocT > _CntrlBlk;
            __cntrl_ = new _CntrlBlk(__r.get(), _VSTD::ref(__r.get_deleter()), _AllocT());
            __enable_weak_this(__r.get(), __r.get());
        }
        __r.release();
    }

    _LIBCPP_HIDE_FROM_ABI
    ~shared_ptr()
    {
        if (__cntrl_)
            __cntrl_->__release_shared();
    }

    _LIBCPP_HIDE_FROM_ABI
    shared_ptr<_Tp>& operator=(const shared_ptr& __r) _NOEXCEPT
    {
        shared_ptr(__r).swap(*this);
        return *this;
    }

    template<class _Yp, class = __enable_if_t<__compatible_with<_Yp, _Tp>::value> >
    _LIBCPP_HIDE_FROM_ABI
    shared_ptr<_Tp>& operator=(const shared_ptr<_Yp>& __r) _NOEXCEPT
    {
        shared_ptr(__r).swap(*this);
        return *this;
    }

    _LIBCPP_HIDE_FROM_ABI
    shared_ptr<_Tp>& operator=(shared_ptr&& __r) _NOEXCEPT
    {
        shared_ptr(_VSTD::move(__r)).swap(*this);
        return *this;
    }

    template<class _Yp, class = __enable_if_t<__compatible_with<_Yp, _Tp>::value> >
    _LIBCPP_HIDE_FROM_ABI
    shared_ptr<_Tp>& operator=(shared_ptr<_Yp>&& __r)
    {
        shared_ptr(_VSTD::move(__r)).swap(*this);
        return *this;
    }

#if _LIBCPP_STD_VER <= 14 || defined(_LIBCPP_ENABLE_CXX17_REMOVED_AUTO_PTR)
    template<class _Yp, class = __enable_if_t<
        !is_array<_Yp>::value &&
        is_convertible<_Yp*, typename shared_ptr<_Tp>::element_type*>::value
    > >
    _LIBCPP_HIDE_FROM_ABI
    shared_ptr<_Tp>& operator=(auto_ptr<_Yp>&& __r)
    {
        shared_ptr(_VSTD::move(__r)).swap(*this);
        return *this;
    }
#endif

    template <class _Yp, class _Dp, class = __enable_if_t<
        is_convertible<typename unique_ptr<_Yp, _Dp>::pointer, element_type*>::value
    > >
    _LIBCPP_HIDE_FROM_ABI
    shared_ptr<_Tp>& operator=(unique_ptr<_Yp, _Dp>&& __r)
    {
        shared_ptr(_VSTD::move(__r)).swap(*this);
        return *this;
    }

    _LIBCPP_HIDE_FROM_ABI
    void swap(shared_ptr& __r) _NOEXCEPT
    {
        _VSTD::swap(__ptr_, __r.__ptr_);
        _VSTD::swap(__cntrl_, __r.__cntrl_);
    }

    _LIBCPP_HIDE_FROM_ABI
    void reset() _NOEXCEPT
    {
        shared_ptr().swap(*this);
    }

    template<class _Yp, class = __enable_if_t<
        __compatible_with<_Yp, _Tp>::value
    > >
    _LIBCPP_HIDE_FROM_ABI
    void reset(_Yp* __p)
    {
        shared_ptr(__p).swap(*this);
    }

    template<class _Yp, class _Dp, class = __enable_if_t<
        __compatible_with<_Yp, _Tp>::value
    > >
    _LIBCPP_HIDE_FROM_ABI
    void reset(_Yp* __p, _Dp __d)
    {
        shared_ptr(__p, __d).swap(*this);
    }

    template<class _Yp, class _Dp, class _Alloc, class = __enable_if_t<
        __compatible_with<_Yp, _Tp>::value
    > >
    _LIBCPP_HIDE_FROM_ABI
    void reset(_Yp* __p, _Dp __d, _Alloc __a)
    {
        shared_ptr(__p, __d, __a).swap(*this);
    }

    _LIBCPP_HIDE_FROM_ABI
    element_type* get() const _NOEXCEPT
    {
        return __ptr_;
    }

    _LIBCPP_HIDE_FROM_ABI
    typename add_lvalue_reference<element_type>::type operator*() const _NOEXCEPT
    {
        return *__ptr_;
    }

    _LIBCPP_HIDE_FROM_ABI
    element_type* operator->() const _NOEXCEPT
    {
        static_assert(!is_array<_Tp>::value,
                      "std::shared_ptr<T>::operator-> is only valid when T is not an array type.");
        return __ptr_;
    }

    _LIBCPP_HIDE_FROM_ABI
    long use_count() const _NOEXCEPT
    {
        return __cntrl_ ? __cntrl_->use_count() : 0;
    }

    _LIBCPP_HIDE_FROM_ABI
    bool unique() const _NOEXCEPT
    {
        return use_count() == 1;
    }

    _LIBCPP_HIDE_FROM_ABI
    explicit operator bool() const _NOEXCEPT
    {
        return get() != nullptr;
    }

    template <class _Up>
    _LIBCPP_HIDE_FROM_ABI
    bool owner_before(shared_ptr<_Up> const& __p) const _NOEXCEPT
    {
        return __cntrl_ < __p.__cntrl_;
    }

    template <class _Up>
    _LIBCPP_HIDE_FROM_ABI
    bool owner_before(weak_ptr<_Up> const& __p) const _NOEXCEPT
    {
        return __cntrl_ < __p.__cntrl_;
    }

    _LIBCPP_HIDE_FROM_ABI
    bool __owner_equivalent(const shared_ptr& __p) const
    {
        return __cntrl_ == __p.__cntrl_;
    }

#if _LIBCPP_STD_VER > 14
    _LIBCPP_HIDE_FROM_ABI
    typename add_lvalue_reference<element_type>::type operator[](ptrdiff_t __i) const
    {
            static_assert(is_array<_Tp>::value,
                          "std::shared_ptr<T>::operator[] is only valid when T is an array type.");
            return __ptr_[__i];
    }
#endif

#ifndef _LIBCPP_NO_RTTI
    template <class _Dp>
    _LIBCPP_HIDE_FROM_ABI
    _Dp* __get_deleter() const _NOEXCEPT
    {
        return static_cast<_Dp*>(__cntrl_
                    ? const_cast<void *>(__cntrl_->__get_deleter(typeid(_Dp)))
                      : nullptr);
    }
#endif // _LIBCPP_NO_RTTI

    template<class _Yp, class _CntrlBlk>
    _LIBCPP_HIDE_FROM_ABI
    static shared_ptr<_Tp> __create_with_control_block(_Yp* __p, _CntrlBlk* __cntrl) _NOEXCEPT
    {
        shared_ptr<_Tp> __r;
        __r.__ptr_ = __p;
        __r.__cntrl_ = __cntrl;
        __r.__enable_weak_this(__r.__ptr_, __r.__ptr_);
        return __r;
    }

private:
    template <class _Yp, bool = is_function<_Yp>::value>
    struct __shared_ptr_default_allocator
    {
        typedef allocator<_Yp> type;
    };

    template <class _Yp>
    struct __shared_ptr_default_allocator<_Yp, true>
    {
        typedef allocator<__shared_ptr_dummy_rebind_allocator_type> type;
    };

    template <class _Yp, class _OrigPtr, class = __enable_if_t<
        is_convertible<_OrigPtr*, const enable_shared_from_this<_Yp>*>::value
    > >
    _LIBCPP_HIDE_FROM_ABI
    void __enable_weak_this(const enable_shared_from_this<_Yp>* __e, _OrigPtr* __ptr) _NOEXCEPT
    {
        typedef typename remove_cv<_Yp>::type _RawYp;
        if (__e && __e->__weak_this_.expired())
        {
            __e->__weak_this_ = shared_ptr<_RawYp>(*this,
                const_cast<_RawYp*>(static_cast<const _Yp*>(__ptr)));
        }
    }

    _LIBCPP_HIDE_FROM_ABI void __enable_weak_this(...) _NOEXCEPT { }

    template <class, class _Yp>
    struct __shared_ptr_default_delete
        : default_delete<_Yp>
    { };

    template <class _Yp, class _Un, size_t _Sz>
    struct __shared_ptr_default_delete<_Yp[_Sz], _Un>
        : default_delete<_Yp[]>
    { };

    template <class _Yp, class _Un>
    struct __shared_ptr_default_delete<_Yp[], _Un>
        : default_delete<_Yp[]>
    { };

    template <class _Up> friend class _LIBCPP_TEMPLATE_VIS shared_ptr;
    template <class _Up> friend class _LIBCPP_TEMPLATE_VIS weak_ptr;
};

#if _LIBCPP_STD_VER > 14
template<class _Tp>
shared_ptr(weak_ptr<_Tp>) -> shared_ptr<_Tp>;
template<class _Tp, class _Dp>
shared_ptr(unique_ptr<_Tp, _Dp>) -> shared_ptr<_Tp>;
#endif

//
// std::allocate_shared and std::make_shared
//
template<class _Tp, class _Alloc, class ..._Args, class = __enable_if_t<!is_array<_Tp>::value> >
_LIBCPP_HIDE_FROM_ABI
shared_ptr<_Tp> allocate_shared(const _Alloc& __a, _Args&& ...__args)
{
    using _ControlBlock = __shared_ptr_emplace<_Tp, _Alloc>;
    using _ControlBlockAllocator = typename __allocator_traits_rebind<_Alloc, _ControlBlock>::type;
    __allocation_guard<_ControlBlockAllocator> __guard(__a, 1);
    ::new ((void*)_VSTD::addressof(*__guard.__get())) _ControlBlock(__a, _VSTD::forward<_Args>(__args)...);
    auto __control_block = __guard.__release_ptr();
    return shared_ptr<_Tp>::__create_with_control_block((*__control_block).__get_elem(), _VSTD::addressof(*__control_block));
}

template<class _Tp, class ..._Args, class = __enable_if_t<!is_array<_Tp>::value> >
_LIBCPP_HIDE_FROM_ABI
shared_ptr<_Tp> make_shared(_Args&& ...__args)
{
    return _VSTD::allocate_shared<_Tp>(allocator<_Tp>(), _VSTD::forward<_Args>(__args)...);
}

template<class _Tp, class _Up>
inline _LIBCPP_INLINE_VISIBILITY
bool
operator==(const shared_ptr<_Tp>& __x, const shared_ptr<_Up>& __y) _NOEXCEPT
{
    return __x.get() == __y.get();
}

template<class _Tp, class _Up>
inline _LIBCPP_INLINE_VISIBILITY
bool
operator!=(const shared_ptr<_Tp>& __x, const shared_ptr<_Up>& __y) _NOEXCEPT
{
    return !(__x == __y);
}

template<class _Tp, class _Up>
inline _LIBCPP_INLINE_VISIBILITY
bool
operator<(const shared_ptr<_Tp>& __x, const shared_ptr<_Up>& __y) _NOEXCEPT
{
#if _LIBCPP_STD_VER <= 11
    typedef typename common_type<_Tp*, _Up*>::type _Vp;
    return less<_Vp>()(__x.get(), __y.get());
#else
    return less<>()(__x.get(), __y.get());
#endif

}

template<class _Tp, class _Up>
inline _LIBCPP_INLINE_VISIBILITY
bool
operator>(const shared_ptr<_Tp>& __x, const shared_ptr<_Up>& __y) _NOEXCEPT
{
    return __y < __x;
}

template<class _Tp, class _Up>
inline _LIBCPP_INLINE_VISIBILITY
bool
operator<=(const shared_ptr<_Tp>& __x, const shared_ptr<_Up>& __y) _NOEXCEPT
{
    return !(__y < __x);
}

template<class _Tp, class _Up>
inline _LIBCPP_INLINE_VISIBILITY
bool
operator>=(const shared_ptr<_Tp>& __x, const shared_ptr<_Up>& __y) _NOEXCEPT
{
    return !(__x < __y);
}

template<class _Tp>
inline _LIBCPP_INLINE_VISIBILITY
bool
operator==(const shared_ptr<_Tp>& __x, nullptr_t) _NOEXCEPT
{
    return !__x;
}

template<class _Tp>
inline _LIBCPP_INLINE_VISIBILITY
bool
operator==(nullptr_t, const shared_ptr<_Tp>& __x) _NOEXCEPT
{
    return !__x;
}

template<class _Tp>
inline _LIBCPP_INLINE_VISIBILITY
bool
operator!=(const shared_ptr<_Tp>& __x, nullptr_t) _NOEXCEPT
{
    return static_cast<bool>(__x);
}

template<class _Tp>
inline _LIBCPP_INLINE_VISIBILITY
bool
operator!=(nullptr_t, const shared_ptr<_Tp>& __x) _NOEXCEPT
{
    return static_cast<bool>(__x);
}

template<class _Tp>
inline _LIBCPP_INLINE_VISIBILITY
bool
operator<(const shared_ptr<_Tp>& __x, nullptr_t) _NOEXCEPT
{
    return less<_Tp*>()(__x.get(), nullptr);
}

template<class _Tp>
inline _LIBCPP_INLINE_VISIBILITY
bool
operator<(nullptr_t, const shared_ptr<_Tp>& __x) _NOEXCEPT
{
    return less<_Tp*>()(nullptr, __x.get());
}

template<class _Tp>
inline _LIBCPP_INLINE_VISIBILITY
bool
operator>(const shared_ptr<_Tp>& __x, nullptr_t) _NOEXCEPT
{
    return nullptr < __x;
}

template<class _Tp>
inline _LIBCPP_INLINE_VISIBILITY
bool
operator>(nullptr_t, const shared_ptr<_Tp>& __x) _NOEXCEPT
{
    return __x < nullptr;
}

template<class _Tp>
inline _LIBCPP_INLINE_VISIBILITY
bool
operator<=(const shared_ptr<_Tp>& __x, nullptr_t) _NOEXCEPT
{
    return !(nullptr < __x);
}

template<class _Tp>
inline _LIBCPP_INLINE_VISIBILITY
bool
operator<=(nullptr_t, const shared_ptr<_Tp>& __x) _NOEXCEPT
{
    return !(__x < nullptr);
}

template<class _Tp>
inline _LIBCPP_INLINE_VISIBILITY
bool
operator>=(const shared_ptr<_Tp>& __x, nullptr_t) _NOEXCEPT
{
    return !(__x < nullptr);
}

template<class _Tp>
inline _LIBCPP_INLINE_VISIBILITY
bool
operator>=(nullptr_t, const shared_ptr<_Tp>& __x) _NOEXCEPT
{
    return !(nullptr < __x);
}

template<class _Tp>
inline _LIBCPP_INLINE_VISIBILITY
void
swap(shared_ptr<_Tp>& __x, shared_ptr<_Tp>& __y) _NOEXCEPT
{
    __x.swap(__y);
}

template<class _Tp, class _Up>
inline _LIBCPP_INLINE_VISIBILITY
shared_ptr<_Tp>
static_pointer_cast(const shared_ptr<_Up>& __r) _NOEXCEPT
{
    return shared_ptr<_Tp>(__r,
                           static_cast<
                               typename shared_ptr<_Tp>::element_type*>(__r.get()));
}

template<class _Tp, class _Up>
inline _LIBCPP_INLINE_VISIBILITY
shared_ptr<_Tp>
dynamic_pointer_cast(const shared_ptr<_Up>& __r) _NOEXCEPT
{
    typedef typename shared_ptr<_Tp>::element_type _ET;
    _ET* __p = dynamic_cast<_ET*>(__r.get());
    return __p ? shared_ptr<_Tp>(__r, __p) : shared_ptr<_Tp>();
}

template<class _Tp, class _Up>
shared_ptr<_Tp>
const_pointer_cast(const shared_ptr<_Up>& __r) _NOEXCEPT
{
    typedef typename shared_ptr<_Tp>::element_type _RTp;
    return shared_ptr<_Tp>(__r, const_cast<_RTp*>(__r.get()));
}

template<class _Tp, class _Up>
shared_ptr<_Tp>
reinterpret_pointer_cast(const shared_ptr<_Up>& __r) _NOEXCEPT
{
    return shared_ptr<_Tp>(__r,
                           reinterpret_cast<
                               typename shared_ptr<_Tp>::element_type*>(__r.get()));
}

#ifndef _LIBCPP_NO_RTTI

template<class _Dp, class _Tp>
inline _LIBCPP_INLINE_VISIBILITY
_Dp*
get_deleter(const shared_ptr<_Tp>& __p) _NOEXCEPT
{
    return __p.template __get_deleter<_Dp>();
}

#endif // _LIBCPP_NO_RTTI

template<class _Tp>
class _LIBCPP_SHARED_PTR_TRIVIAL_ABI _LIBCPP_TEMPLATE_VIS weak_ptr
{
public:
#if _LIBCPP_STD_VER > 14
    typedef remove_extent_t<_Tp> element_type;
#else
    typedef _Tp element_type;
#endif

private:
    element_type*        __ptr_;
    __shared_weak_count* __cntrl_;

public:
    _LIBCPP_INLINE_VISIBILITY
    _LIBCPP_CONSTEXPR weak_ptr() _NOEXCEPT;
    template<class _Yp> _LIBCPP_INLINE_VISIBILITY weak_ptr(shared_ptr<_Yp> const& __r,
                   typename enable_if<__compatible_with<_Yp, _Tp>::value, __nat*>::type = 0)
                        _NOEXCEPT;
    _LIBCPP_INLINE_VISIBILITY
    weak_ptr(weak_ptr const& __r) _NOEXCEPT;
    template<class _Yp> _LIBCPP_INLINE_VISIBILITY weak_ptr(weak_ptr<_Yp> const& __r,
                   typename enable_if<__compatible_with<_Yp, _Tp>::value, __nat*>::type = 0)
                         _NOEXCEPT;

    _LIBCPP_INLINE_VISIBILITY
    weak_ptr(weak_ptr&& __r) _NOEXCEPT;
    template<class _Yp> _LIBCPP_INLINE_VISIBILITY weak_ptr(weak_ptr<_Yp>&& __r,
                   typename enable_if<__compatible_with<_Yp, _Tp>::value, __nat*>::type = 0)
                         _NOEXCEPT;
    ~weak_ptr();

    _LIBCPP_INLINE_VISIBILITY
    weak_ptr& operator=(weak_ptr const& __r) _NOEXCEPT;
    template<class _Yp>
        typename enable_if
        <
            __compatible_with<_Yp, _Tp>::value,
            weak_ptr&
        >::type
        _LIBCPP_INLINE_VISIBILITY
        operator=(weak_ptr<_Yp> const& __r) _NOEXCEPT;

    _LIBCPP_INLINE_VISIBILITY
    weak_ptr& operator=(weak_ptr&& __r) _NOEXCEPT;
    template<class _Yp>
        typename enable_if
        <
            __compatible_with<_Yp, _Tp>::value,
            weak_ptr&
        >::type
        _LIBCPP_INLINE_VISIBILITY
        operator=(weak_ptr<_Yp>&& __r) _NOEXCEPT;

    template<class _Yp>
        typename enable_if
        <
            __compatible_with<_Yp, _Tp>::value,
            weak_ptr&
        >::type
        _LIBCPP_INLINE_VISIBILITY
        operator=(shared_ptr<_Yp> const& __r) _NOEXCEPT;

    _LIBCPP_INLINE_VISIBILITY
    void swap(weak_ptr& __r) _NOEXCEPT;
    _LIBCPP_INLINE_VISIBILITY
    void reset() _NOEXCEPT;

    _LIBCPP_INLINE_VISIBILITY
    long use_count() const _NOEXCEPT
        {return __cntrl_ ? __cntrl_->use_count() : 0;}
    _LIBCPP_INLINE_VISIBILITY
    bool expired() const _NOEXCEPT
        {return __cntrl_ == nullptr || __cntrl_->use_count() == 0;}
    shared_ptr<_Tp> lock() const _NOEXCEPT;
    template<class _Up>
        _LIBCPP_INLINE_VISIBILITY
        bool owner_before(const shared_ptr<_Up>& __r) const _NOEXCEPT
        {return __cntrl_ < __r.__cntrl_;}
    template<class _Up>
        _LIBCPP_INLINE_VISIBILITY
        bool owner_before(const weak_ptr<_Up>& __r) const _NOEXCEPT
        {return __cntrl_ < __r.__cntrl_;}

    template <class _Up> friend class _LIBCPP_TEMPLATE_VIS weak_ptr;
    template <class _Up> friend class _LIBCPP_TEMPLATE_VIS shared_ptr;
};

#if _LIBCPP_STD_VER > 14
template<class _Tp>
weak_ptr(shared_ptr<_Tp>) -> weak_ptr<_Tp>;
#endif

template<class _Tp>
inline
_LIBCPP_CONSTEXPR
weak_ptr<_Tp>::weak_ptr() _NOEXCEPT
    : __ptr_(nullptr),
      __cntrl_(nullptr)
{
}

template<class _Tp>
inline
weak_ptr<_Tp>::weak_ptr(weak_ptr const& __r) _NOEXCEPT
    : __ptr_(__r.__ptr_),
      __cntrl_(__r.__cntrl_)
{
    if (__cntrl_)
        __cntrl_->__add_weak();
}

template<class _Tp>
template<class _Yp>
inline
weak_ptr<_Tp>::weak_ptr(shared_ptr<_Yp> const& __r,
                        typename enable_if<__compatible_with<_Yp, _Tp>::value, __nat*>::type)
                         _NOEXCEPT
    : __ptr_(__r.__ptr_),
      __cntrl_(__r.__cntrl_)
{
    if (__cntrl_)
        __cntrl_->__add_weak();
}

template<class _Tp>
template<class _Yp>
inline
weak_ptr<_Tp>::weak_ptr(weak_ptr<_Yp> const& __r,
                        typename enable_if<__compatible_with<_Yp, _Tp>::value, __nat*>::type)
         _NOEXCEPT
    : __ptr_(__r.__ptr_),
      __cntrl_(__r.__cntrl_)
{
    if (__cntrl_)
        __cntrl_->__add_weak();
}

template<class _Tp>
inline
weak_ptr<_Tp>::weak_ptr(weak_ptr&& __r) _NOEXCEPT
    : __ptr_(__r.__ptr_),
      __cntrl_(__r.__cntrl_)
{
    __r.__ptr_ = nullptr;
    __r.__cntrl_ = nullptr;
}

template<class _Tp>
template<class _Yp>
inline
weak_ptr<_Tp>::weak_ptr(weak_ptr<_Yp>&& __r,
                        typename enable_if<__compatible_with<_Yp, _Tp>::value, __nat*>::type)
         _NOEXCEPT
    : __ptr_(__r.__ptr_),
      __cntrl_(__r.__cntrl_)
{
    __r.__ptr_ = nullptr;
    __r.__cntrl_ = nullptr;
}

template<class _Tp>
weak_ptr<_Tp>::~weak_ptr()
{
    if (__cntrl_)
        __cntrl_->__release_weak();
}

template<class _Tp>
inline
weak_ptr<_Tp>&
weak_ptr<_Tp>::operator=(weak_ptr const& __r) _NOEXCEPT
{
    weak_ptr(__r).swap(*this);
    return *this;
}

template<class _Tp>
template<class _Yp>
inline
typename enable_if
<
    __compatible_with<_Yp, _Tp>::value,
    weak_ptr<_Tp>&
>::type
weak_ptr<_Tp>::operator=(weak_ptr<_Yp> const& __r) _NOEXCEPT
{
    weak_ptr(__r).swap(*this);
    return *this;
}

template<class _Tp>
inline
weak_ptr<_Tp>&
weak_ptr<_Tp>::operator=(weak_ptr&& __r) _NOEXCEPT
{
    weak_ptr(_VSTD::move(__r)).swap(*this);
    return *this;
}

template<class _Tp>
template<class _Yp>
inline
typename enable_if
<
    __compatible_with<_Yp, _Tp>::value,
    weak_ptr<_Tp>&
>::type
weak_ptr<_Tp>::operator=(weak_ptr<_Yp>&& __r) _NOEXCEPT
{
    weak_ptr(_VSTD::move(__r)).swap(*this);
    return *this;
}

template<class _Tp>
template<class _Yp>
inline
typename enable_if
<
    __compatible_with<_Yp, _Tp>::value,
    weak_ptr<_Tp>&
>::type
weak_ptr<_Tp>::operator=(shared_ptr<_Yp> const& __r) _NOEXCEPT
{
    weak_ptr(__r).swap(*this);
    return *this;
}

template<class _Tp>
inline
void
weak_ptr<_Tp>::swap(weak_ptr& __r) _NOEXCEPT
{
    _VSTD::swap(__ptr_, __r.__ptr_);
    _VSTD::swap(__cntrl_, __r.__cntrl_);
}

template<class _Tp>
inline _LIBCPP_INLINE_VISIBILITY
void
swap(weak_ptr<_Tp>& __x, weak_ptr<_Tp>& __y) _NOEXCEPT
{
    __x.swap(__y);
}

template<class _Tp>
inline
void
weak_ptr<_Tp>::reset() _NOEXCEPT
{
    weak_ptr().swap(*this);
}

template<class _Tp>
shared_ptr<_Tp>
weak_ptr<_Tp>::lock() const _NOEXCEPT
{
    shared_ptr<_Tp> __r;
    __r.__cntrl_ = __cntrl_ ? __cntrl_->lock() : __cntrl_;
    if (__r.__cntrl_)
        __r.__ptr_ = __ptr_;
    return __r;
}

#if _LIBCPP_STD_VER > 14
template <class _Tp = void> struct owner_less;
#else
template <class _Tp> struct owner_less;
#endif


_LIBCPP_SUPPRESS_DEPRECATED_PUSH
template <class _Tp>
struct _LIBCPP_TEMPLATE_VIS owner_less<shared_ptr<_Tp> >
#if !defined(_LIBCPP_ABI_NO_BINDER_BASES)
    : binary_function<shared_ptr<_Tp>, shared_ptr<_Tp>, bool>
#endif
{
_LIBCPP_SUPPRESS_DEPRECATED_POP
#if _LIBCPP_STD_VER <= 17 || defined(_LIBCPP_ENABLE_CXX20_REMOVED_BINDER_TYPEDEFS)
    _LIBCPP_DEPRECATED_IN_CXX17 typedef bool result_type;
    _LIBCPP_DEPRECATED_IN_CXX17 typedef shared_ptr<_Tp> first_argument_type;
    _LIBCPP_DEPRECATED_IN_CXX17 typedef shared_ptr<_Tp> second_argument_type;
#endif
    _LIBCPP_INLINE_VISIBILITY
    bool operator()(shared_ptr<_Tp> const& __x, shared_ptr<_Tp> const& __y) const _NOEXCEPT
        {return __x.owner_before(__y);}
    _LIBCPP_INLINE_VISIBILITY
    bool operator()(shared_ptr<_Tp> const& __x,   weak_ptr<_Tp> const& __y) const _NOEXCEPT
        {return __x.owner_before(__y);}
    _LIBCPP_INLINE_VISIBILITY
    bool operator()(  weak_ptr<_Tp> const& __x, shared_ptr<_Tp> const& __y) const _NOEXCEPT
        {return __x.owner_before(__y);}
};

_LIBCPP_SUPPRESS_DEPRECATED_PUSH
template <class _Tp>
struct _LIBCPP_TEMPLATE_VIS owner_less<weak_ptr<_Tp> >
#if !defined(_LIBCPP_ABI_NO_BINDER_BASES)
    : binary_function<weak_ptr<_Tp>, weak_ptr<_Tp>, bool>
#endif
{
_LIBCPP_SUPPRESS_DEPRECATED_POP
#if _LIBCPP_STD_VER <= 17 || defined(_LIBCPP_ENABLE_CXX20_REMOVED_BINDER_TYPEDEFS)
    _LIBCPP_DEPRECATED_IN_CXX17 typedef bool result_type;
    _LIBCPP_DEPRECATED_IN_CXX17 typedef weak_ptr<_Tp> first_argument_type;
    _LIBCPP_DEPRECATED_IN_CXX17 typedef weak_ptr<_Tp> second_argument_type;
#endif
    _LIBCPP_INLINE_VISIBILITY
    bool operator()(  weak_ptr<_Tp> const& __x,   weak_ptr<_Tp> const& __y) const _NOEXCEPT
        {return __x.owner_before(__y);}
    _LIBCPP_INLINE_VISIBILITY
    bool operator()(shared_ptr<_Tp> const& __x,   weak_ptr<_Tp> const& __y) const _NOEXCEPT
        {return __x.owner_before(__y);}
    _LIBCPP_INLINE_VISIBILITY
    bool operator()(  weak_ptr<_Tp> const& __x, shared_ptr<_Tp> const& __y) const _NOEXCEPT
        {return __x.owner_before(__y);}
};

#if _LIBCPP_STD_VER > 14
template <>
struct _LIBCPP_TEMPLATE_VIS owner_less<void>
{
    template <class _Tp, class _Up>
    _LIBCPP_INLINE_VISIBILITY
    bool operator()( shared_ptr<_Tp> const& __x, shared_ptr<_Up> const& __y) const _NOEXCEPT
        {return __x.owner_before(__y);}
    template <class _Tp, class _Up>
    _LIBCPP_INLINE_VISIBILITY
    bool operator()( shared_ptr<_Tp> const& __x,   weak_ptr<_Up> const& __y) const _NOEXCEPT
        {return __x.owner_before(__y);}
    template <class _Tp, class _Up>
    _LIBCPP_INLINE_VISIBILITY
    bool operator()(   weak_ptr<_Tp> const& __x, shared_ptr<_Up> const& __y) const _NOEXCEPT
        {return __x.owner_before(__y);}
    template <class _Tp, class _Up>
    _LIBCPP_INLINE_VISIBILITY
    bool operator()(   weak_ptr<_Tp> const& __x,   weak_ptr<_Up> const& __y) const _NOEXCEPT
        {return __x.owner_before(__y);}
    typedef void is_transparent;
};
#endif

template<class _Tp>
class _LIBCPP_TEMPLATE_VIS enable_shared_from_this
{
    mutable weak_ptr<_Tp> __weak_this_;
protected:
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR
    enable_shared_from_this() _NOEXCEPT {}
    _LIBCPP_INLINE_VISIBILITY
    enable_shared_from_this(enable_shared_from_this const&) _NOEXCEPT {}
    _LIBCPP_INLINE_VISIBILITY
    enable_shared_from_this& operator=(enable_shared_from_this const&) _NOEXCEPT
        {return *this;}
    _LIBCPP_INLINE_VISIBILITY
    ~enable_shared_from_this() {}
public:
    _LIBCPP_INLINE_VISIBILITY
    shared_ptr<_Tp> shared_from_this()
        {return shared_ptr<_Tp>(__weak_this_);}
    _LIBCPP_INLINE_VISIBILITY
    shared_ptr<_Tp const> shared_from_this() const
        {return shared_ptr<const _Tp>(__weak_this_);}

#if _LIBCPP_STD_VER > 14
    _LIBCPP_INLINE_VISIBILITY
    weak_ptr<_Tp> weak_from_this() _NOEXCEPT
       { return __weak_this_; }

    _LIBCPP_INLINE_VISIBILITY
    weak_ptr<const _Tp> weak_from_this() const _NOEXCEPT
        { return __weak_this_; }
#endif // _LIBCPP_STD_VER > 14

    template <class _Up> friend class shared_ptr;
};

template <class _Tp> struct _LIBCPP_TEMPLATE_VIS hash;

template <class _Tp>
struct _LIBCPP_TEMPLATE_VIS hash<shared_ptr<_Tp> >
{
#if _LIBCPP_STD_VER <= 17 || defined(_LIBCPP_ENABLE_CXX20_REMOVED_BINDER_TYPEDEFS)
    _LIBCPP_DEPRECATED_IN_CXX17 typedef shared_ptr<_Tp> argument_type;
    _LIBCPP_DEPRECATED_IN_CXX17 typedef size_t          result_type;
#endif

    _LIBCPP_INLINE_VISIBILITY
    size_t operator()(const shared_ptr<_Tp>& __ptr) const _NOEXCEPT
    {
        return hash<typename shared_ptr<_Tp>::element_type*>()(__ptr.get());
    }
};

template<class _CharT, class _Traits, class _Yp>
inline _LIBCPP_INLINE_VISIBILITY
basic_ostream<_CharT, _Traits>&
operator<<(basic_ostream<_CharT, _Traits>& __os, shared_ptr<_Yp> const& __p);


#if !defined(_LIBCPP_HAS_NO_THREADS)

class _LIBCPP_TYPE_VIS __sp_mut
{
    void* __lx;
public:
    void lock() _NOEXCEPT;
    void unlock() _NOEXCEPT;

private:
    _LIBCPP_CONSTEXPR __sp_mut(void*) _NOEXCEPT;
    __sp_mut(const __sp_mut&);
    __sp_mut& operator=(const __sp_mut&);

    friend _LIBCPP_FUNC_VIS __sp_mut& __get_sp_mut(const void*);
};

_LIBCPP_FUNC_VIS _LIBCPP_AVAILABILITY_ATOMIC_SHARED_PTR
__sp_mut& __get_sp_mut(const void*);

template <class _Tp>
inline _LIBCPP_INLINE_VISIBILITY
bool
atomic_is_lock_free(const shared_ptr<_Tp>*)
{
    return false;
}

template <class _Tp>
_LIBCPP_AVAILABILITY_ATOMIC_SHARED_PTR
shared_ptr<_Tp>
atomic_load(const shared_ptr<_Tp>* __p)
{
    __sp_mut& __m = __get_sp_mut(__p);
    __m.lock();
    shared_ptr<_Tp> __q = *__p;
    __m.unlock();
    return __q;
}

template <class _Tp>
inline _LIBCPP_INLINE_VISIBILITY
_LIBCPP_AVAILABILITY_ATOMIC_SHARED_PTR
shared_ptr<_Tp>
atomic_load_explicit(const shared_ptr<_Tp>* __p, memory_order)
{
    return atomic_load(__p);
}

template <class _Tp>
_LIBCPP_AVAILABILITY_ATOMIC_SHARED_PTR
void
atomic_store(shared_ptr<_Tp>* __p, shared_ptr<_Tp> __r)
{
    __sp_mut& __m = __get_sp_mut(__p);
    __m.lock();
    __p->swap(__r);
    __m.unlock();
}

template <class _Tp>
inline _LIBCPP_INLINE_VISIBILITY
_LIBCPP_AVAILABILITY_ATOMIC_SHARED_PTR
void
atomic_store_explicit(shared_ptr<_Tp>* __p, shared_ptr<_Tp> __r, memory_order)
{
    atomic_store(__p, __r);
}

template <class _Tp>
_LIBCPP_AVAILABILITY_ATOMIC_SHARED_PTR
shared_ptr<_Tp>
atomic_exchange(shared_ptr<_Tp>* __p, shared_ptr<_Tp> __r)
{
    __sp_mut& __m = __get_sp_mut(__p);
    __m.lock();
    __p->swap(__r);
    __m.unlock();
    return __r;
}

template <class _Tp>
inline _LIBCPP_INLINE_VISIBILITY
_LIBCPP_AVAILABILITY_ATOMIC_SHARED_PTR
shared_ptr<_Tp>
atomic_exchange_explicit(shared_ptr<_Tp>* __p, shared_ptr<_Tp> __r, memory_order)
{
    return atomic_exchange(__p, __r);
}

template <class _Tp>
_LIBCPP_AVAILABILITY_ATOMIC_SHARED_PTR
bool
atomic_compare_exchange_strong(shared_ptr<_Tp>* __p, shared_ptr<_Tp>* __v, shared_ptr<_Tp> __w)
{
    shared_ptr<_Tp> __temp;
    __sp_mut& __m = __get_sp_mut(__p);
    __m.lock();
    if (__p->__owner_equivalent(*__v))
    {
        _VSTD::swap(__temp, *__p);
        *__p = __w;
        __m.unlock();
        return true;
    }
    _VSTD::swap(__temp, *__v);
    *__v = *__p;
    __m.unlock();
    return false;
}

template <class _Tp>
inline _LIBCPP_INLINE_VISIBILITY
_LIBCPP_AVAILABILITY_ATOMIC_SHARED_PTR
bool
atomic_compare_exchange_weak(shared_ptr<_Tp>* __p, shared_ptr<_Tp>* __v, shared_ptr<_Tp> __w)
{
    return atomic_compare_exchange_strong(__p, __v, __w);
}

template <class _Tp>
inline _LIBCPP_INLINE_VISIBILITY
_LIBCPP_AVAILABILITY_ATOMIC_SHARED_PTR
bool
atomic_compare_exchange_strong_explicit(shared_ptr<_Tp>* __p, shared_ptr<_Tp>* __v,
                                        shared_ptr<_Tp> __w, memory_order, memory_order)
{
    return atomic_compare_exchange_strong(__p, __v, __w);
}

template <class _Tp>
inline _LIBCPP_INLINE_VISIBILITY
_LIBCPP_AVAILABILITY_ATOMIC_SHARED_PTR
bool
atomic_compare_exchange_weak_explicit(shared_ptr<_Tp>* __p, shared_ptr<_Tp>* __v,
                                      shared_ptr<_Tp> __w, memory_order, memory_order)
{
    return atomic_compare_exchange_weak(__p, __v, __w);
}

#endif // !defined(_LIBCPP_HAS_NO_THREADS)

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___MEMORY_SHARED_PTR_H
