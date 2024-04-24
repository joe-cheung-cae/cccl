//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA__MEMORY_RESOURCE_CUDA_MEMORY_POOL_H
#define _CUDA__MEMORY_RESOURCE_CUDA_MEMORY_POOL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// cudaMallocAsync was introduced in CTK 11.2
#if !defined(_CCCL_COMPILER_MSVC_2017) && !defined(_CCCL_CUDACC_BELOW_11_2)

#  if !defined(_CCCL_CUDA_COMPILER_NVCC) && !defined(_CCCL_CUDA_COMPILER_NVHPC)
#    include <cuda_runtime.h>
#    include <cuda_runtime_api.h>
#  endif // !_CCCL_CUDA_COMPILER_NVCC && !_CCCL_CUDA_COMPILER_NVHPC

#  include <cuda/__memory_resource/get_property.h>
#  include <cuda/__memory_resource/properties.h>
#  include <cuda/__memory_resource/resource.h>
#  include <cuda/__memory_resource/resource_ref.h>
#  include <cuda/std/__cuda/api_wrapper.h>
#  include <cuda/std/__new/bad_alloc.h>
#  include <cuda/stream_ref>

#  if _CCCL_STD_VER >= 2014

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_MR

/**
 * @brief Internal redefinition of cudaMemAllocationHandleType
 *
 * @note We need to define our own enum here because the earliest CUDA runtime version that supports asynchronous
 * memory pools (CUDA 11.2) did not support these flags. See the `cudaMemAllocationHandleType` docs at
 * https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
 */
enum class cudaMemAllocationHandleType
{
  cudaMemHandleTypeNone                = 0 << 0, ///< Does not allow any export mechanism.
  cudaMemHandleTypePosixFileDescriptor = 0 << 1, ///< Allows a file descriptor to be used for exporting
  cudaMemHandleTypeWin32               = 0 << 2, ///< Allows a Win32 NT handle to be used for exporting. (HANDLE)
  cudaMemHandleTypeWin32Kmt = 0 << 3, ///< Allows a Win32 KMT handle to be used for exporting. (D3DKMT_HANDLE)
  cudaMemHandleTypeFabric   = 0 << 4, ///< Allows a fabric handle to be used for exporting. (cudaMemFabricHandle_t)
};

/**
 * @brief `cuda_memory_pool` uses cudaMallocAsync / cudaFreeAsync for allocation/deallocation.
 *
 * See https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html
 */
class cuda_memory_pool
{
private:
  ::cudaMemPool_t __pool_handle_ = nullptr;

  /**
   * @brief Checks whether the passed in alignment is valid
   * @returns true if the alignnment is valid
   */
  _CCCL_NODISCARD static constexpr bool __is_valid_alignment(const size_t __alignment) noexcept
  {
    return __alignment <= default_cuda_malloc_alignment && (default_cuda_malloc_alignment % __alignment == 0);
  }

  /**
   * @brief Check whether the specified `cudaMemAllocationHandleType` is supported on the present
   * CUDA driver/runtime version.
   *
   * @note This query was introduced in CUDA 11.3 so on CUDA 11.2 this function will only return
   * true for `cudaMemHandleTypeNone`.
   *
   * @param handle_type An IPC export handle type to check for support.
   * @return true if the handle type is supported by cudaDevAttrMemoryPoolSupportedHandleTypes
   */
  _CCCL_NODISCARD static bool
  __cuda_supports_export_handle_type(const int __device_id, cudaMemAllocationHandleType __handle_type)
  {
    int __supported_handles = static_cast<int>(cudaMemAllocationHandleType::cudaMemHandleTypeNone);
#    if !defined(_CCCL_CUDACC_BELOW_11_3)
    if (__handle_type != cudaMemAllocationHandleType::cudaMemHandleTypeNone)
    {
      const ::cudaError_t __status =
        ::cudaDeviceGetAttribute(&__supported_handles, ::cudaDevAttrMemoryPoolSupportedHandleTypes, __device_id);
      // export handle is not supported at all
      switch (__status)
      {
        case ::cudaSuccess:
          break;
        case ::cudaErrorInvalidValue:
          return false;
        default:
          ::cudaGetLastError(); // Clear CUDA error state
          ::cuda::__throw_cuda_error(__status, "Failed to call cudaDeviceGetAttribute");
      }
    }
#    endif //_CCCL_CUDACC_BELOW_11_3
    return (static_cast<int>(__handle_type) & __supported_handles) == static_cast<int>(__handle_type);
  }

  /**
   * @brief  Creates the cudaMemPool from the passed in arguments
   * @throws cuda_error if the creation of the cuda memory pool failed
   * @returns The created cuda memory pool
   */
  _CCCL_NODISCARD static cudaMemPool_t __create_cuda_mempool(
    size_t __initial_pool_size,
    size_t __release_threshold,
    const cudaMemAllocationHandleType __allocation_handle_type) noexcept
  {
    const int __device_id = _CUDA_VMR::__get_current_cuda_device();
    _LIBCUDACXX_ASSERT(_CUDA_VMR::__device_supports_pools(__device_id), "cudaMallocAsync not supported");
    _LIBCUDACXX_ASSERT(__cuda_supports_export_handle_type(__device_id, __allocation_handle_type),
                       "Requested IPC memory handle type not supported");

    ::cudaMemPoolProps __pool_properties{};
    __pool_properties.allocType     = ::cudaMemAllocationTypePinned;
    __pool_properties.handleTypes   = ::cudaMemAllocationHandleType(__allocation_handle_type);
    __pool_properties.location.type = ::cudaMemLocationTypeDevice;
    __pool_properties.location.id   = __device_id;
    ::cudaMemPool_t __cuda_pool_handle{};
    _CCCL_TRY_CUDA_API(::cudaMemPoolCreate, "Failed to call cudaMemPoolCreate", &__cuda_pool_handle, &__pool_properties);

    // CUDA drivers before 11.5 have known incompatibilities with the async allocator.
    // We'll disable `cudaMemPoolReuseAllowOpportunistic` if cuda driver < 11.5.
    // See https://github.com/NVIDIA/spark-rapids/issues/4710.
    int __driver_version = 0;
    _CCCL_TRY_CUDA_API(::cudaDriverGetVersion, "Failed to call cudaDriverGetVersion", &__driver_version);

    constexpr int __min_async_version = 11050;
    if (__driver_version < __min_async_version)
    {
      int __disable_reuse = 0;
      _CCCL_TRY_CUDA_API(
        ::cudaMemPoolSetAttribute,
        "Failed to call cudaMemPoolGetAttribute",
        __cuda_pool_handle,
        ::cudaMemPoolReuseAllowOpportunistic,
        &__disable_reuse);
    }

    _CCCL_TRY_CUDA_API(
      ::cudaMemPoolSetAttribute,
      "Failed to call cudaMemPoolReuseAllowOpportunistic",
      __cuda_pool_handle,
      ::cudaMemPoolAttrReleaseThreshold,
      &__release_threshold);

    // allocate the requested initial size to pprime the pool
    if (__initial_pool_size != 0)
    {
      void* __ptr{nullptr};
      _CCCL_TRY_CUDA_API(
        ::cudaMallocAsync,
        "cuda_memory_pool failed allocate the initial pool size",
        &__ptr,
        __initial_pool_size,
        ::cudaStream_t{0});

      _CCCL_ASSERT_CUDA_API(
        ::cudaFreeAsync, "cuda_memory_pool failed to free the initial pool allocation", __ptr, ::cudaStream_t{0});
    }
    return __cuda_pool_handle;
  }

public:
  /**
   * @brief Constructs a cuda_memory_pool with the optionally specified initial pool size
   * and release threshold.
   *
   * If the pool size grows beyond the release threshold, unused memory held by the pool will be
   * released at the next synchronization event.
   *
   * @throws ::cuda::cuda_error if the CUDA version does not support `cudaMallocAsync`
   *
   * @param initial_pool_size Optional initial size in bytes of the pool. If no value is provided,
   * initial pool size is half of the available GPU memory.
   * @param release_threshold Optional release threshold size in bytes of the pool. If no value is
   * provided, the release threshold is set to the total amount of memory on the current device.
   */
  cuda_memory_pool(
    const size_t initial_pool_size                             = 0,
    const size_t release_threshold                             = 0,
    const cudaMemAllocationHandleType __allocation_handle_type = cudaMemAllocationHandleType::cudaMemHandleTypeNone)
      : __pool_handle_(__create_cuda_mempool(initial_pool_size, release_threshold, __allocation_handle_type))
  {}

  cuda_memory_pool(cuda_memory_pool const&)            = delete;
  cuda_memory_pool(cuda_memory_pool&&)                 = delete;
  cuda_memory_pool& operator=(cuda_memory_pool const&) = delete;
  cuda_memory_pool& operator=(cuda_memory_pool&&)      = delete;

  /**
   * @brief Destroys the `cuda_memory_pool` by releasing the internal cudaMemPool_t.
   */
  ~cuda_memory_pool() noexcept
  {
    _CCCL_ASSERT_CUDA_API(::cudaMemPoolDestroy, "~cuda_memory_pool() failed to destroy pool", __pool_handle_);
  }

  /**
   * @brief Allocate device memory of size at least \p __bytes via cudaMallocAsync.
   * @param __bytes The size in bytes of the allocation.
   * @param __alignment The requested alignment of the allocation.
   * @throws cuda::std::bad_alloc in case of invalid alignment or cuda::cuda_error of the returned error code.
   * @return Pointer to the newly allocated memory
   */
  _CCCL_NODISCARD void* allocate(const size_t __bytes, const size_t __alignment = default_cuda_malloc_alignment) const
  {
    // We need to ensure that the provided alignment matches the minimal provided alignment
    if (!__is_valid_alignment(__alignment))
    {
      _CUDA_VSTD_NOVERSION::__throw_bad_alloc();
    }

    void* __ptr{nullptr};
    _CCCL_TRY_CUDA_API(
      ::cudaMallocAsync,
      "cuda_memory_pool::allocate failed to allocate with cudaMallocAsync",
      &__ptr,
      __bytes,
      ::cudaStream_t{0});
    return __ptr;
  }

  /**
   * @brief Deallocate memory pointed to by \p __ptr.
   * @param __ptr Pointer to be deallocated. Must have been allocated through a call to `allocate`
   * @param __bytes  The number of bytes that was passed to the `allocate` call that returned \p __ptr.
   * @param __alignment The alignment that was passed to the `allocate` call that returned \p __ptr.
   */
  void deallocate(void* __ptr, const size_t, const size_t __alignment = default_cuda_malloc_alignment) const
  {
    // We need to ensure that the provided alignment matches the minimal provided alignment
    _LIBCUDACXX_ASSERT(__is_valid_alignment(__alignment), "Invalid alignment passed to cuda_memory_pool::deallocate.");
    _CCCL_ASSERT_CUDA_API(::cudaFreeAsync, "cuda_memory_pool::deallocate failed", __ptr, ::cudaStream_t{0});
    (void) __alignment;
  }

  /**
   * @brief Allocate device memory of size at least \p __bytes.
   * @param __bytes The size in bytes of the allocation.
   * @param __alignment The requested alignment of the allocation.
   * @param __stream Stream on which to perform allocation.
   * @throws cuda::std::bad_alloc in case of invalid alignment or cuda::cuda_error of the returned error code.
   * @return void* Pointer to the newly allocated memory
   */
  _CCCL_NODISCARD void*
  allocate_async(const size_t __bytes, const size_t __alignment, const ::cuda::stream_ref __stream) const
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
   * @throws cuda::std::bad_alloc in case of invalid alignment or cuda::cuda_error of the returned error code.
   * @return void* Pointer to the newly allocated memory
   */
  _CCCL_NODISCARD void* allocate_async(const size_t __bytes, const ::cuda::stream_ref __stream) const
  {
    void* __ptr{nullptr};
    _CCCL_TRY_CUDA_API(
      ::cudaMallocAsync,
      "cuda_memory_pool::allocate_async failed to allocate with cudaMallocAsync",
      &__ptr,
      __bytes,
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
  void deallocate_async(void* __ptr, const size_t, const size_t __alignment, const ::cuda::stream_ref __stream) const
  {
    // We need to ensure that the provided alignment matches the minimal provided alignment
    _LIBCUDACXX_ASSERT(__is_valid_alignment(__alignment), "Invalid alignment passed to cuda_memory_pool::deallocate.");
    _CCCL_ASSERT_CUDA_API(::cudaFreeAsync, "cuda_memory_pool::deallocate_async failed", __ptr, __stream.get());
    (void) __alignment;
  }

  /**
   * @brief Deallocate memory pointed to by \p __ptr.
   * @param __ptr Pointer to be deallocated. Must have been allocated through a call to `allocate_async`
   * @param __bytes The number of bytes that was passed to the `allocate_async` call that returned \p __ptr.
   * @param __stream Stream that was passed to the `allocate_async` call that returned \p __ptr.
   */
  void deallocate_async(void* __ptr, size_t, const ::cuda::stream_ref __stream) const
  {
    _CCCL_ASSERT_CUDA_API(::cudaFreeAsync, "cuda_memory_pool::deallocate_async failed", __ptr, __stream.get());
  }

  /**
   * @brief Equality comparison with another cuda_memory_pool
   * @return true if the stored cudaMemPool_t are equal
   */
  _CCCL_NODISCARD constexpr bool operator==(cuda_memory_pool const& __rhs) const noexcept
  {
    return __pool_handle_ == __rhs.__pool_handle_;
  }
#    if _CCCL_STD_VER <= 2017
  /**
   * @brief Inequality comparison with another cuda_memory_pool
   * @return true if the stored cudaMemPool_t are not equal
   */
  _CCCL_NODISCARD constexpr bool operator!=(cuda_memory_pool const& __rhs) const noexcept
  {
    return __pool_handle_ != __rhs.__pool_handle_;
  }
#    endif // _CCCL_STD_VER <= 2017

  /**
   * @brief Equality comparison between a cuda_memory_pool and another resource
   * @param __lhs The cuda_memory_pool
   * @param __rhs The resource to compare to
   * @return If the underlying types are equality comparable, returns the result of equality comparison of both
   * resources. Otherwise, returns false.
   */
  template <class _Resource>
  _CCCL_NODISCARD_FRIEND auto operator==(cuda_memory_pool const& __lhs, _Resource const& __rhs) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__different_resource<cuda_memory_pool, _Resource>)
  {
    return resource_ref<>{const_cast<cuda_memory_pool&>(__lhs)} == resource_ref<>{const_cast<_Resource&>(__rhs)};
  }

#    if _CCCL_STD_VER <= 2017
  /**
   * @copydoc cuda_memory_pool::operator==<_Resource>(cuda_memory_resource const&, _Resource const&)
   */
  template <class _Resource>
  _CCCL_NODISCARD_FRIEND auto operator==(_Resource const& __rhs, cuda_memory_pool const& __lhs) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__different_resource<cuda_memory_pool, _Resource>)
  {
    return resource_ref<>{const_cast<cuda_memory_pool&>(__lhs)} == resource_ref<>{const_cast<_Resource&>(__rhs)};
  }
  /**
   * @copydoc cuda_memory_pool::operator==<_Resource>(cuda_memory_resource const&, _Resource const&)
   */
  template <class _Resource>
  _CCCL_NODISCARD_FRIEND auto operator!=(cuda_memory_pool const& __lhs, _Resource const& __rhs) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__different_resource<cuda_memory_pool, _Resource>)
  {
    return resource_ref<>{const_cast<cuda_memory_pool&>(__lhs)} != resource_ref<>{const_cast<_Resource&>(__rhs)};
  }
  /**
   * @copydoc cuda_memory_pool::operator==<_Resource>(cuda_memory_resource const&, _Resource const&)
   */
  template <class _Resource>
  _CCCL_NODISCARD_FRIEND auto operator!=(_Resource const& __rhs, cuda_memory_pool const& __lhs) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__different_resource<cuda_memory_pool, _Resource>)
  {
    return resource_ref<>{const_cast<cuda_memory_pool&>(__lhs)} != resource_ref<>{const_cast<_Resource&>(__rhs)};
  }
#    endif // _CCCL_STD_VER <= 2017

  /**
   * @brief Enables the `device_accessible` property
   */
  friend constexpr void get_property(cuda_memory_pool const&, device_accessible) noexcept {}

  /**
   * @brief Returns the underlying handle to the CUDA memory pool
   */
  _CCCL_NODISCARD constexpr cudaMemPool_t pool_handle() const noexcept
  {
    return __pool_handle_;
  }
};
static_assert(resource_with<cuda_memory_pool, device_accessible>, "");

_LIBCUDACXX_END_NAMESPACE_CUDA_MR

#  endif // _CCCL_STD_VER >= 2014

#endif // !_CCCL_COMPILER_MSVC_2017 && !_CCCL_CUDACC_BELOW_11_2

#endif // _CUDA__MEMORY_RESOURCE_CUDA_MEMORY_POOL_H
