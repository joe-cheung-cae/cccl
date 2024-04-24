//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA__STD__CUDA_API_WRAPPER_H
#define _CUDA__STD__CUDA_API_WRAPPER_H

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

#include <cuda/std/__exception/cuda_error.h>

#define _CCCL_TRY_CUDA_API(_NAME, _MSG, ...)           \
  {                                                    \
    const ::cudaError_t __status = _NAME(__VA_ARGS__); \
    switch (__status)                                  \
    {                                                  \
      case ::cudaSuccess:                              \
        break;                                         \
      default:                                         \
        ::cudaGetLastError();                          \
        ::cuda::__throw_cuda_error(__status, _MSG);    \
    }                                                  \
  }

#define _CCCL_ASSERT_CUDA_API(_NAME, _MSG, ...)        \
  {                                                    \
    const ::cudaError_t __status = _NAME(__VA_ARGS__); \
    _LIBCUDACXX_ASSERT(__status == cudaSuccess, _MSG); \
    (void) __status;                                   \
  }

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_MR

/**
 * @brief  Returns the device id of the current device
 * @throws cuda_error if cudaGetDevice was not successful
 * @returns The device id
 */
_CCCL_NODISCARD static int __get_current_cuda_device()
{
  int __device = -1;
  _CCCL_TRY_CUDA_API(::cudaGetDevice, "Failed to query current device with cudaGetDevice.", &__device);
  return __device;
}

/**
 * @brief  Checks whether the current device supports cudaMallocAsync
 * @param __device_id The device id of the current device
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
 * @param __device_id The device id of the current device
 * @throws cuda_error if retrieving the default cudaMemPool_t fails
 * @returns The default memory pool of the current device
 */
_CCCL_NODISCARD static cudaMemPool_t __get_default_mem_pool(const int __device_id)
{
  _LIBCUDACXX_ASSERT(_CUDA_VMR::__device_supports_pools(__device_id), "cudaMallocAsync not supported");

  ::cudaMemPool_t __pool;
  _CCCL_TRY_CUDA_API(::cudaDeviceGetDefaultMemPool, "Failed to call cudaDeviceGetDefaultMemPool", &__pool, __device_id);
  return __pool;
}

_LIBCUDACXX_END_NAMESPACE_CUDA_MR

#endif //_CUDA__STD__CUDA_API_WRAPPER_H
