//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA__MEMORY_RESOURCE_CUDA_ASYNC_MEMORY_RESOURCE_H
#define _CUDA__MEMORY_RESOURCE_CUDA_ASYNC_MEMORY_RESOURCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// cudaMallocAsync was introduced in CTK 11.2
#if !defined(_CCCL_COMPILER_MSVC_2017) && defined(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE) \
  && !defined(_CCCL_CUDACC_BELOW_11_2)

#  if !defined(_CCCL_CUDA_COMPILER_NVCC) && !defined(_CCCL_CUDA_COMPILER_NVHPC)
#    include <cuda_runtime.h>
#    include <cuda_runtime_api.h>
#  endif // !_CCCL_CUDA_COMPILER_NVCC && !_CCCL_CUDA_COMPILER_NVHPC

#  include <cuda/__memory_resource/cuda_memory_pool.h>
#  include <cuda/__memory_resource/get_property.h>
#  include <cuda/__memory_resource/properties.h>
#  include <cuda/__memory_resource/resource.h>
#  include <cuda/__memory_resource/resource_ref.h>
#  include <cuda/std/__concepts/__concept_macros.h>
#  include <cuda/std/__cuda/api_wrapper.h>
#  include <cuda/std/__memory/addressof.h>
#  include <cuda/std/__new/bad_alloc.h>
#  include <cuda/stream_ref>

#  if _CCCL_STD_VER >= 2014

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_MR

/**
 * @brief `cuda_async_memory_resource` uses cudaMallocFromPoolAsync / cudaFreeAsync for allocation/deallocation.
 *
 * See https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html
 *
 * A `cuda_async_memory_resource` is a thin wrapper around a `cudaMemPool_t`. It does not own the pool and it is the
 * responsibility of the user to ensure that the lifetime of the pool exceeds the lifetime of this
 * `cuda_async_memory_resource`.
 */
class cuda_async_memory_resource
{
private:
  ::cudaMemPool_t __pool_;

  /**
   * @brief Checks whether the passed in alignment is valid
   * @returns true if the alignnment is valid
   */
  _CCCL_NODISCARD static constexpr bool __is_valid_alignment(const size_t __alignment) noexcept
  {
    return __alignment <= default_cuda_malloc_alignment && (default_cuda_malloc_alignment % __alignment == 0);
  }

public:
  /**
   * @brief  Constructs the cuda_async_memory_resource from a cudaMemPool_t. If none is provided it uses the default
   * cudaMemPool_t of the current device
   * @throws cuda_error if retrieving the default cudaMemPool_t fails
   */
  cuda_async_memory_resource(
    ::cudaMemPool_t __pool = _CUDA_VMR::__get_default_mem_pool(_CUDA_VMR::__get_current_cuda_device()))
      : __pool_(__pool)
  {}

  /**
   * @brief  Constructs the cuda_async_memory_resource from a cuda_memory_pool by calling pool_handle()
   */
  cuda_async_memory_resource(cuda_memory_pool& __cuda_pool) noexcept
      : __pool_(__cuda_pool.pool_handle())
  {}

  /**
   * @brief Allocate device memory of size at least \p __bytes via cudaMallocFromPoolAsync.
   * @param __bytes The size in bytes of the allocation.
   * @param __alignment The requested alignment of the allocation.
   * @throws cuda::std::bad_alloc in case of invalid alignment or cuda::cuda_error of the returned error code.
   * @return Pointer to the newly allocated memory
   */
  _CCCL_NODISCARD void* allocate(const size_t __bytes, const size_t __alignment = default_cuda_malloc_alignment)
  {
    // We need to ensure that the provided alignment matches the minimal provided alignment
    if (!__is_valid_alignment(__alignment))
    {
      _CUDA_VSTD_NOVERSION::__throw_bad_alloc();
    }

    void* __ptr{nullptr};
    _CCCL_TRY_CUDA_API(
      ::cudaMallocFromPoolAsync,
      "cuda_async_memory_resource::allocate failed to allocate with cudaMallocFromPoolAsync",
      &__ptr,
      __bytes,
      __pool_,
      ::cudaStream_t{0});
    return __ptr;
  }

  /**
   * @brief Deallocate memory pointed to by \p __ptr.
   * @param __ptr Pointer to be deallocated. Must have been allocated through a call to `allocate`
   * @param __bytes  The number of bytes that was passed to the `allocate` call that returned \p __ptr.
   * @param __alignment The alignment that was passed to the `allocate` call that returned \p __ptr.
   */
  void deallocate(void* __ptr, const size_t, const size_t __alignment = default_cuda_malloc_alignment)
  {
    // We need to ensure that the provided alignment matches the minimal provided alignment
    _LIBCUDACXX_ASSERT(__is_valid_alignment(__alignment),
                       "Invalid alignment passed to cuda_memory_resource::deallocate.");
    _CCCL_ASSERT_CUDA_API(::cudaFreeAsync, "cuda_async_memory_resource::deallocate failed", __ptr, ::cudaStream_t{0});
    (void) __alignment;
  }

  /**
   * @brief Allocate device memory of size at least \p __bytes via cudaMallocFromPoolAsync.
   * @param __bytes The size in bytes of the allocation.
   * @param __alignment The requested alignment of the allocation.
   * @param __stream Stream on which to perform allocation.
   * @throws cuda::cuda_error of the returned error code.
   * @return void* Pointer to the newly allocated memory
   */
  _CCCL_NODISCARD void* allocate_async(const size_t __bytes, const size_t __alignment, const ::cuda::stream_ref __stream)
  {
    // We need to ensure that the provided alignment matches the minimal provided alignment
    if (!__is_valid_alignment(__alignment))
    {
      _CUDA_VSTD_NOVERSION::__throw_bad_alloc();
    }

    return allocate_async(__bytes, __stream);
  }

  /**
   * @brief Allocate device memory of size at least \p __bytes via cudaMallocFromPoolAsync.
   * @param __bytes The size in bytes of the allocation.
   * @param __stream Stream on which to perform allocation.
   * @throws cuda::cuda_error of the returned error code.
   * @return void* Pointer to the newly allocated memory
   */
  _CCCL_NODISCARD void* allocate_async(const size_t __bytes, const ::cuda::stream_ref __stream)
  {
    void* __ptr{nullptr};
    _CCCL_TRY_CUDA_API(
      ::cudaMallocFromPoolAsync,
      "cuda_async_memory_resource::allocate_async failed to allocate with cudaMallocFromPoolAsync",
      &__ptr,
      __bytes,
      __pool_,
      __stream.get());
    return __ptr;
  }

  /**
   * @brief Deallocate memory pointed to by \p __ptr.
   * @param __ptr Pointer to be deallocated. Must have been allocated through a call to `allocate_async`
   * @param __bytes The number of bytes that was passed to the `allocate_async` call that returned \p __ptr.
   * @param __alignment The alignment that was passed to the `allocate_async` call that returned \p __ptr.
   * @param __stream A stream that has a stream ordering relationship with the stream used in the `allocate_async` call
   * that returned \p __ptr. See https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html
   */
  void deallocate_async(void* __ptr, const size_t __bytes, const size_t __alignment, const ::cuda::stream_ref __stream)
  {
    // We need to ensure that the provided alignment matches the minimal provided alignment
    _LIBCUDACXX_ASSERT(__is_valid_alignment(__alignment),
                       "Invalid alignment passed to cuda_memory_resource::deallocate.");
    deallocate_async(__ptr, __bytes, __stream);
    (void) __alignment;
  }

  /**
   * @brief Deallocate memory pointed to by \p __ptr.
   * @param __ptr Pointer to be deallocated. Must have been allocated through a call to `allocate_async`
   * @param __bytes The number of bytes that was passed to the `allocate_async` call that returned \p __ptr.
   * @param __stream A stream that has a stream ordering relationship with the stream used in the `allocate_async` call
   * that returned \p __ptr. See https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html
   */
  void deallocate_async(void* __ptr, size_t, const ::cuda::stream_ref __stream)
  {
    _CCCL_ASSERT_CUDA_API(::cudaFreeAsync, "cuda_async_memory_resource::deallocate_async failed", __ptr, __stream.get());
  }

  /**
   * @brief Equality comparison with another cuda_async_memory_resource
   * @return true if underlying cudaMemPool_t are equal
   */
  _CCCL_NODISCARD constexpr bool operator==(cuda_async_memory_resource const& __rhs) const noexcept
  {
    return __pool_ == __rhs.__pool_;
  }
#    if _CCCL_STD_VER <= 2017
  /**
   * @brief Inequality comparison with another cuda_async_memory_resource
   * @return true if underlying cudaMemPool_t are inequal
   */
  _CCCL_NODISCARD constexpr bool operator!=(cuda_async_memory_resource const& __rhs) const noexcept
  {
    return __pool_ != __rhs.__pool_;
  }
#    endif // _CCCL_STD_VER <= 2017

  /**
   * @brief Equality comparison between a cuda_memory_resource and another resource
   * @param __lhs The cuda_memory_resource
   * @param __rhs The resource to compare to
   * @return If the underlying types are equality comparable, returns the result of equality comparison of both
   * resources. Otherwise, returns false.
   */
  template <class _Resource>
  _CCCL_NODISCARD_FRIEND auto operator==(cuda_async_memory_resource const& __lhs, _Resource const& __rhs) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__different_resource<cuda_async_memory_resource, _Resource>)
  {
    return resource_ref<>{const_cast<cuda_async_memory_resource&>(__lhs)}
        == resource_ref<>{const_cast<_Resource&>(__rhs)};
  }
#    if _CCCL_STD_VER <= 2017
  /**
   * @copydoc cuda_async_memory_resource::operator==<_Resource>(cuda_memory_resource const&, _Resource const&)
   */
  template <class _Resource>
  _CCCL_NODISCARD_FRIEND auto operator==(_Resource const& __rhs, cuda_async_memory_resource const& __lhs) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__different_resource<cuda_async_memory_resource, _Resource>)
  {
    return resource_ref<>{const_cast<cuda_async_memory_resource&>(__lhs)}
        == resource_ref<>{const_cast<_Resource&>(__rhs)};
  }
  /**
   * @copydoc cuda_async_memory_resource::operator==<_Resource>(cuda_memory_resource const&, _Resource const&)
   */
  template <class _Resource>
  _CCCL_NODISCARD_FRIEND auto operator!=(cuda_async_memory_resource const& __lhs, _Resource const& __rhs) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__different_resource<cuda_async_memory_resource, _Resource>)
  {
    return resource_ref<>{const_cast<cuda_async_memory_resource&>(__lhs)}
        != resource_ref<>{const_cast<_Resource&>(__rhs)};
  }
  /**
   * @copydoc cuda_async_memory_resource::operator==<_Resource>(cuda_memory_resource const&, _Resource const&)
   */
  template <class _Resource>
  _CCCL_NODISCARD_FRIEND auto operator!=(_Resource const& __rhs, cuda_async_memory_resource const& __lhs) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__different_resource<cuda_async_memory_resource, _Resource>)
  {
    return resource_ref<>{const_cast<cuda_async_memory_resource&>(__lhs)}
        != resource_ref<>{const_cast<_Resource&>(__rhs)};
  }
#    endif // _CCCL_STD_VER <= 2017

  /**
   * @brief Returns the underlying handle to the CUDA memory pool
   */
  _CCCL_NODISCARD constexpr cudaMemPool_t pool_handle() const noexcept
  {
    return __pool_;
  }

  /**
   * @brief Enables the `device_accessible` property
   */
  friend constexpr void get_property(cuda_async_memory_resource const&, device_accessible) noexcept {}
};
static_assert(resource_with<cuda_async_memory_resource, device_accessible>, "");

_LIBCUDACXX_END_NAMESPACE_CUDA_MR

#  endif // _CCCL_STD_VER >= 2014

#endif // !_CCCL_COMPILER_MSVC_2017 && !_CCCL_CUDACC_BELOW_11_2

#endif //_CUDA__MEMORY_RESOURCE_CUDA_ASYNC_MEMORY_RESOURCE_H
