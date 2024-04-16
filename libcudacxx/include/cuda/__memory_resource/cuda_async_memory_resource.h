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

#if !defined(_CCCL_CUDA_COMPILER_NVCC) && !defined(_CCCL_CUDA_COMPILER_NVHPC)
#  include <cuda_runtime_api.h>
#endif // !_CCCL_CUDA_COMPILER_NVCC && !_CCCL_CUDA_COMPILER_NVHPC

#include <cuda/__memory_resource/cuda_memory_pool.h>
#include <cuda/__memory_resource/get_property.h>
#include <cuda/__memory_resource/properties.h>
#include <cuda/__memory_resource/resource.h>
#include <cuda/__memory_resource/resource_ref.h>
#include <cuda/std/__cuda/api_wrapper.h>
#include <cuda/std/detail/libcxx/include/__new/bad_alloc.h>
#include <cuda/stream_ref>

// cudaMallocAsync was introduced in CTK 11.2
#if !defined(_CCCL_COMPILER_MSVC_2017) && !defined(_CCCL_CUDACC_BELOW_11_2)

#  if _CCCL_STD_VER >= 2014

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_MR

/**
 * @brief `cuda_async_memory_resource` uses cudaMallocFromPoolAsync / cudaFreeAsync for allocation/deallocation.
 *
 * See https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html
 *
 * A `cuda_async_memory_resource` can either hold a `cudaMemPool_t` or it can hold its own
 * `cuda_memory_pool` to use with cudaMallocAsync.
 *
 * There are different ways of constructing a `cuda_async_memory_resource`:
 *
 *   1. A default constructed `cuda_async_memory_resource` will hold the current default pool of the current device as
 *      obtained by `cudaDeviceGetDefaultMemPool`. The pool will not be released through `cudaMemPoolDestroy` on
 *      destruction. It is the responsibility of the user to ensure that the lifetime of the pool exceeds the lifetime
 *      of this `cuda_async_memory_resource`.
 *   2. A `cudaMemPool_t` can be provided to the constructor of `cuda_async_memory_resource`. The pool will not be
 *      released through `cudaMemPoolDestroy` on destruction. It is the responsibility of the user to ensure that the
 *      lifetime of the pool exceeds the lifetime of this `cuda_async_memory_resource`.
 *   3. If the initial pool size and optionally the release threshold and optionally the cudaMemAllocationHandleType is
 *      provided, `cuda_async_memory_resource` will construct its own `cuda_memory_pool`. The pool will be released
 *      through `cudaMemPoolDestroy` on destruction.
 *
 */
class cuda_async_memory_resource
{
private:
  union
  {
    ::cudaMemPool_t __provided_pool_;
    cuda_memory_pool __owned_pool_;
  };
  bool __use_provided_pool_;

  /**
   * @brief Checks whether the passed in alignment is valid
   * @returns true if the alignnment is valid
   */
  _CCCL_NODISCARD static constexpr bool __is_valid_alignment(const size_t __alignment) noexcept
  {
    return __alignment <= default_cuda_malloc_alignment && (default_cuda_malloc_alignment % __alignment == 0);
  }

  /**
   * @brief  Returns the device id for the current device
   * @throws cuda_error if cudaGetDevice was not successful
   * @returns The device id
   */
  _CCCL_NODISCARD static int __get_current_cuda_device()
  {
    int __device = -1;
    _CCCL_TRY_CUDA_API(::cudaGetDevice, "Failed to querry current device with with cudaGetDevice.", &__device);
    return __device;
  }

  /**
   * @brief  Checks whether the current device supports cudaMallocAsync
   * @throws cuda_error if cudaDeviceGetAttribute failed
   * @returns true if cudaDevAttrMemoryPoolsSupported is not zero
   */
  _CCCL_NODISCARD static bool __device_supports_pools(const int __device_id)
  {
    int __pool_is_supported = 0;
    _CCCL_TRY_CUDA_API(
      ::cudaDeviceGetAttribute,
      "Failed to call cudaDeviceGetAttribute",
      &__pool_is_supported,
      ::cudaDevAttrMemoryPoolsSupported,
      __device_id);
    return __pool_is_supported != 0;
  }

  /**
   * @brief  Returns the default cudaMemPool_t from the current device
   * @throws cuda_error if retrieving the default cudaMemPool_t fails
   * @returns The default memory pool of the current device
   */
  _CCCL_NODISCARD static cudaMemPool_t __get_default_mem_pool()
  {
    const int __device_id = __get_current_cuda_device();
    _LIBCUDACXX_ASSERT(__device_supports_pools(__device_id), "cudaMallocAsync not supported");

    ::cudaMemPool_t __pool;
    _CCCL_TRY_CUDA_API(
      ::cudaDeviceGetDefaultMemPool, "Failed to call cudaDeviceGetDefaultMemPool", &__pool, __device_id);
    return __pool;
  }

public:
  /**
   * @brief  Constructs the cuda_async_memory_resource ussing the default cudaMemPool_t of the current device
   * @throws cuda_error if retrieving the default cudaMemPool_t fails
   */
  cuda_async_memory_resource(::cudaMemPool_t __provided_pool = __get_default_mem_pool())
      : __provided_pool_(__provided_pool)
      , __use_provided_pool_(true)
  {}

  /**
   * @brief Constructs a cuda_async_memory_resource with the optionally specified initial pool size
   * and release threshold.
   *
   * If the pool size grows beyond the release threshold, unused memory held by the pool will be
   * released at the next synchronization event.
   *
   * @throws ::cuda::cuda_error if the CUDA version does not support `cudaMallocAsync`
   *
   * @param initial_pool_size Initial size in bytes of the pool.
   * @param release_threshold Optional release threshold size in bytes of the pool. If no value is
   * provided, the release threshold is set to the total amount of memory on the current device.
   */
  cuda_async_memory_resource(
    const size_t initial_pool_size,
    const size_t release_threshold                             = 0,
    const cudaMemAllocationHandleType __allocation_handle_type = cudaMemAllocationHandleType::cudaMemHandleTypeNone)
      : __owned_pool_(initial_pool_size, release_threshold, __allocation_handle_type)
      , __use_provided_pool_(false)
  {}

  cuda_async_memory_resource(cuda_async_memory_resource const&)            = delete;
  cuda_async_memory_resource(cuda_async_memory_resource&&)                 = delete;
  cuda_async_memory_resource& operator=(cuda_async_memory_resource const&) = delete;
  cuda_async_memory_resource& operator=(cuda_async_memory_resource&&)      = delete;

  /**
   * @brief Releases the internal cudaMemPool_t if it was not user provided
   */
  ~cuda_async_memory_resource() noexcept
  {
    if (!__use_provided_pool_)
    {
      __owned_pool_.~cuda_memory_pool();
    }
  }

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
      pool_handle(),
      ::cudaStream_t{0});
    return __ptr;
  }

  /**
   * @brief Deallocate memory pointed to by \p __ptr.
   * @param __ptr Pointer to be deallocated. Must have been allocated through a call to `allocate`
   * @param __bytes  The number of bytes that was passed to the `allocate` call that returned \p __ptr.
   * @param __alignment The alignment that was passed to the `allocate` call that returned \p __ptr.
   */
  void deallocate(void* __ptr, const size_t __bytes, const size_t __alignment = default_cuda_malloc_alignment)
  {
    // We need to ensure that the provided alignment matches the minimal provided alignment
    _LIBCUDACXX_ASSERT(__is_valid_alignment(__alignment),
                       "Invalid alignment passed to cuda_memory_resource::deallocate.");
    _CCCL_ASSERT_CUDA_API(::cudaFreeAsync, "cuda_async_memory_resource::deallocate failed", __ptr, ::cudaStream_t{0});
    (void) __alignment;
  }

  /**
   * @brief Allocate device memory of size at least \p __bytes.
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
   * @brief Allocate device memory of size at least \p __bytes.
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
      pool_handle(),
      __stream.get());
    return __ptr;
  }

  /**
   * @brief Deallocate memory pointed to by \p __ptr.
   * @param __ptr Pointer to be deallocated. Must have been allocated through a call to `allocate_async`
   * @param __bytes The number of bytes that was passed to the `allocate_async` call that returned \p __ptr.
   * @param __alignment The alignment that was passed to the `allocate_async` call that returned \p __ptr.
   * @param __stream Stream that was passed to the `allocate_async` call that returned \p __ptr.
   */
  void deallocate_async(void* __ptr, const size_t __bytes, const size_t __alignment, const ::cuda::stream_ref __stream)
  {
    // We need to ensure that the provided alignment matches the minimal provided alignment
    _LIBCUDACXX_ASSERT(__is_valid_alignment(__alignment),
                       "Invalid alignment passed to cuda_memory_resource::deallocate.");
    deallocate_async(__ptr, __bytes, __stream);
  }

  /**
   * @brief Deallocate memory pointed to by \p __ptr.
   * @param __ptr Pointer to be deallocated. Must have been allocated through a call to `allocate_async`
   * @param __bytes The number of bytes that was passed to the `allocate_async` call that returned \p __ptr.
   * @param __stream Stream that was passed to the `allocate_async` call that returned \p __ptr.
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
    return pool_handle() == __rhs.pool_handle();
  }
#    if _CCCL_STD_VER <= 2017
  /**
   * @brief Inequality comparison with another cuda_async_memory_resource
   * @return true if underlying cudaMemPool_t are inequal
   */
  _CCCL_NODISCARD constexpr bool operator!=(cuda_async_memory_resource const& __rhs) const noexcept
  {
    return pool_handle() != __rhs.pool_handle();
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
    return __use_provided_pool_ ? __provided_pool_ : __owned_pool_.pool_handle();
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
