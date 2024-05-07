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
 * @brief  Checks whether the current device supports cudaMallocAsync
 * @param __device_id The device id of the current device
 * @throws cuda_error if cudaDeviceGetAttribute failed
 * @returns true if cudaDevAttrMemoryPoolsSupported is not zero
 */
_CCCL_NODISCARD inline bool __device_supports_pools(const int __device_id)
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
 * @param __device_id The device id of the current device
 * @throws cuda_error if retrieving the default cudaMemPool_t fails
 * @returns The default memory pool of the current device
 */
_CCCL_NODISCARD inline cudaMemPool_t __get_default_mem_pool(const int __device_id)
{
  _LIBCUDACXX_ASSERT(_CUDA_VMR::__device_supports_pools(__device_id), "cudaMallocAsync not supported");

  ::cudaMemPool_t __pool;
  _CCCL_TRY_CUDA_API(::cudaDeviceGetDefaultMemPool, "Failed to call cudaDeviceGetDefaultMemPool", &__pool, __device_id);
  return __pool;
}

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
 * @brief `cuda_memory_pool_properties` is a wrapper around a properties passed to `cuda_memory_pool` to create a cuda
 * memory pool
 *
 * See https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html
 */
struct cuda_memory_pool_properties
{
  size_t initial_pool_size                           = 0;
  size_t release_threshold                           = 0;
  cudaMemAllocationHandleType allocation_handle_type = cudaMemAllocationHandleType::cudaMemHandleTypeNone;
};

/**
 * @brief `cuda_memory_pool` is a simple wrapper around a `cudaMemPool_t`
 *
 * See https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html
 */
class cuda_memory_pool
{
private:
  ::cudaMemPool_t __pool_handle_ = nullptr;

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
  _CCCL_NODISCARD static cudaMemPool_t
  __create_cuda_mempool(const int __device_id, cuda_memory_pool_properties __properties) noexcept
  {
    _LIBCUDACXX_ASSERT(_CUDA_VMR::__device_supports_pools(__device_id), "cudaMallocAsync not supported");
    _LIBCUDACXX_ASSERT(__cuda_supports_export_handle_type(__device_id, __properties.allocation_handle_type),
                       "Requested IPC memory handle type not supported");

    ::cudaMemPoolProps __pool_properties{};
    __pool_properties.allocType     = ::cudaMemAllocationTypePinned;
    __pool_properties.handleTypes   = ::cudaMemAllocationHandleType(__properties.allocation_handle_type);
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
      &__properties.release_threshold);

    // allocate the requested initial size to pprime the pool
    if (__properties.initial_pool_size != 0)
    {
      void* __ptr{nullptr};
      _CCCL_TRY_CUDA_API(
        ::cudaMallocAsync,
        "cuda_memory_pool failed allocate the initial pool size",
        &__ptr,
        __properties.initial_pool_size,
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
   * @param __device_id The device id of the device the stream pool is constructed on.
   * @param __pool_properties Optional, additional properties of the pool to be created
   */
  explicit cuda_memory_pool(const int __device_id, cuda_memory_pool_properties __properties = {})
      : __pool_handle_(__create_cuda_mempool(__device_id, __properties))
  {}

  /**
   * @brief Disables construction from a plain `cudaMemPool_t`. We want to ensure clean ownership semantics
   */
  cuda_memory_pool(::cudaMemPool_t) = delete;

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
   * @brief Equality comparison with a cudaMemPool_t
   * @param __rhs A cudaMemPool_t
   * @return true if the stored cudaMemPool_t is equal to \p __rhs
   */
  _CCCL_NODISCARD_FRIEND constexpr bool operator==(cuda_memory_pool const& __lhs, ::cudaMemPool_t __rhs) noexcept
  {
    return __lhs.__pool_handle_ == __rhs;
  }
#    if _CCCL_STD_VER <= 2017
  /**
   * @copydoc cuda_memory_pool::operator==(cuda_memory_pool const&, ::cudaMemPool_t)
   */
  _CCCL_NODISCARD_FRIEND constexpr bool operator==(::cudaMemPool_t __lhs, cuda_memory_pool const& __rhs) noexcept
  {
    return __rhs.__pool_handle_ == __lhs;
  }
  /**
   * @copydoc cuda_memory_pool::operator==(cuda_memory_pool const&, ::cudaMemPool_t)
   */
  _CCCL_NODISCARD_FRIEND constexpr bool operator!=(cuda_memory_pool const& __lhs, ::cudaMemPool_t __rhs) noexcept
  {
    return __lhs.__pool_handle_ != __rhs;
  }
  /**
   * @copydoc cuda_memory_pool::operator==(cuda_memory_pool const&, ::cudaMemPool_t)
   */
  _CCCL_NODISCARD_FRIEND constexpr bool operator!=(::cudaMemPool_t __lhs, cuda_memory_pool const& __rhs) noexcept
  {
    return __rhs.__pool_handle_ != __lhs;
  }
#    endif // _CCCL_STD_VER <= 2017

  /**
   * @brief Returns the underlying handle to the CUDA memory pool
   */
  _CCCL_NODISCARD constexpr cudaMemPool_t pool_handle() const noexcept
  {
    return __pool_handle_;
  }
};

_LIBCUDACXX_END_NAMESPACE_CUDA_MR

#  endif // _CCCL_STD_VER >= 2014

#endif // !_CCCL_COMPILER_MSVC_2017 && !_CCCL_CUDACC_BELOW_11_2

#endif // _CUDA__MEMORY_RESOURCE_CUDA_MEMORY_POOL_H
