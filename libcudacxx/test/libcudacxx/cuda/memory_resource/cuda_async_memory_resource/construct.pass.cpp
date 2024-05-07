//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: nvcc-11.1
// UNSUPPORTED: !nvcc && clang
// UNSUPPORTED: nvrtc
#define LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE

#include <cuda/memory_resource>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/stream_ref>

#include "test_macros.h"

bool ensure_release_threshold(::cudaMemPool_t pool, const size_t expected_threshold)
{
  size_t release_threshold = expected_threshold + 1337; // use something different than the expected threshold
  _CCCL_TRY_CUDA_API(
    ::cudaMemPoolGetAttribute,
    "Failed to call cudaMemPoolGetAttribute",
    pool,
    ::cudaMemPoolAttrReleaseThreshold,
    &release_threshold);
  return release_threshold == expected_threshold;
}

bool ensure_disable_reuse(::cudaMemPool_t pool, const int driver_version)
{
  int disable_reuse = 0;
  _CCCL_TRY_CUDA_API(
    ::cudaMemPoolGetAttribute,
    "Failed to call cudaMemPoolGetAttribute",
    pool,
    ::cudaMemPoolReuseAllowOpportunistic,
    &disable_reuse);

  constexpr int min_async_version = 11050;
  return driver_version < min_async_version ? disable_reuse == 0 : disable_reuse != 0;
}

bool ensure_export_handle(::cudaMemPool_t pool, const ::cudaMemAllocationHandleType allocation_handle)
{
  size_t handle              = 0;
  const ::cudaError_t status = ::cudaMemPoolExportToShareableHandle(&handle, pool, allocation_handle, 0);
  ::cudaGetLastError(); // Clear CUDA error state

  // If no export was defined we need to query cudaErrorInvalidValue
  return allocation_handle == ::cudaMemHandleTypeNone ? status == ::cudaErrorInvalidValue : status == ::cudaSuccess;
}

void test()
{
  int current_device{};
  {
    _CCCL_TRY_CUDA_API(::cudaGetDevice, "Failed to query current device with cudaGetDevice.", &current_device);
  }

  int driver_version = 0;
  {
    _CCCL_TRY_CUDA_API(::cudaDriverGetVersion, "Failed to call cudaDriverGetVersion", &driver_version);
  }

  ::cudaMemPool_t current_default_pool{};
  {
    _CCCL_TRY_CUDA_API(::cudaDeviceGetDefaultMemPool,
                       "Failed to call cudaDeviceGetDefaultMemPool",
                       &current_default_pool,
                       current_device);
  }

  using async_resource = cuda::mr::cuda_async_memory_resource;
  {
    {
      async_resource default_constructed{};
      assert(default_constructed.pool_handle() == current_default_pool);
    }

    // Ensure that the pool was not destroyed by allocating something
    void* ptr{nullptr};
    _CCCL_TRY_CUDA_API(
      ::cudaMallocAsync,
      "Failed to allocate with pool passed to cuda::mr::cuda_async_memory_resource",
      &ptr,
      42,
      current_default_pool,
      ::cudaStream_t{0});
    assert(ptr != nullptr);

    _CCCL_ASSERT_CUDA_API(
      ::cudaFreeAsync,
      "Failed to deallocate with pool passed to cuda::mr::cuda_async_memory_resource",
      ptr,
      ::cudaStream_t{0});
  }

  {
    ::cudaMemPoolProps pool_properties{};
    pool_properties.allocType     = ::cudaMemAllocationTypePinned;
    pool_properties.handleTypes   = ::cudaMemAllocationHandleType(0);
    pool_properties.location.type = ::cudaMemLocationTypeDevice;
    pool_properties.location.id   = current_device;
    cudaMemPool_t cuda_pool_handle{};
    _CCCL_TRY_CUDA_API(::cudaMemPoolCreate, "Failed to call cudaMemPoolCreate", &cuda_pool_handle, &pool_properties);

    {
      async_resource from_cudaMemPool{cuda_pool_handle};
      assert(from_cudaMemPool.pool_handle() == cuda_pool_handle);
      assert(from_cudaMemPool.pool_handle() != current_default_pool);
    }

    // Ensure that the pool was not destroyed by allocating something
    void* ptr{nullptr};
    _CCCL_TRY_CUDA_API(
      ::cudaMallocAsync,
      "Failed to allocate with pool passed to cuda::mr::cuda_async_memory_resource",
      &ptr,
      42,
      current_default_pool,
      ::cudaStream_t{0});
    assert(ptr != nullptr);

    _CCCL_ASSERT_CUDA_API(
      ::cudaFreeAsync,
      "Failed to deallocate with pool passed to cuda::mr::cuda_async_memory_resource",
      ptr,
      ::cudaStream_t{0});
  }

  {
    cuda::mr::cuda_memory_pool_properties props = {
      42,
    };
    cuda::mr::cuda_memory_pool pool{current_device, props};
    async_resource from_initial_pool_size{pool};

    ::cudaMemPool_t pool_handle = from_initial_pool_size.pool_handle();
    assert(pool_handle != current_default_pool);

    // Ensure we use the right release threshold
    assert(ensure_release_threshold(pool_handle, 0));

    // Ensure that we disable reuse with unsupported drivers
    assert(ensure_disable_reuse(pool_handle, driver_version));

    // Ensure that we disable export
    assert(ensure_export_handle(pool_handle, ::cudaMemHandleTypeNone));
  }

  {
    cuda::mr::cuda_memory_pool_properties props = {
      42,
      20,
    };
    cuda::mr::cuda_memory_pool pool{current_device, props};
    async_resource with_threshold{pool};

    ::cudaMemPool_t pool_handle = with_threshold.pool_handle();
    assert(pool_handle != current_default_pool);

    // Ensure we use the right release threshold
    assert(ensure_release_threshold(pool_handle, props.release_threshold));

    // Ensure that we disable reuse with unsupported drivers
    assert(ensure_disable_reuse(pool_handle, driver_version));

    // Ensure that we disable export
    assert(ensure_export_handle(pool_handle, ::cudaMemHandleTypeNone));
  }

  // Allocation handles are only supported after 11.2
#if !defined(TEST_COMPILER_CUDACC_BELOW_11_3)
  {
    cuda::mr::cuda_memory_pool_properties props = {
      42,
      20,
      cuda::mr::cudaMemAllocationHandleType::cudaMemHandleTypePosixFileDescriptor,
    };
    cuda::mr::cuda_memory_pool pool{current_device, props};
    async_resource with_allocation_handle{pool};

    ::cudaMemPool_t pool_handle = with_allocation_handle.pool_handle();
    assert(pool_handle != current_default_pool);

    // Ensure we use the right release threshold
    assert(ensure_release_threshold(pool_handle, props.release_threshold));

    // Ensure that we disable reuse with unsupported drivers
    assert(ensure_disable_reuse(pool_handle, driver_version));

    // Ensure that we disable export
    assert(ensure_export_handle(pool_handle, static_cast<cudaMemAllocationHandleType>(props.allocation_handle_type)));
  }
#endif // !TEST_COMPILER_CUDACC_BELOW_11_3
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, test();)
  return 0;
}
