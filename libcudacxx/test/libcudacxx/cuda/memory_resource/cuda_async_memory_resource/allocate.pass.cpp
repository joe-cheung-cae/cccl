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

void ensure_device_ptr(void* ptr)
{
  assert(ptr != nullptr);
  cudaPointerAttributes attributes;
  cudaError_t status = cudaPointerGetAttributes(&attributes, ptr);
  assert(status == cudaSuccess);
  assert(attributes.type == cudaMemoryTypeDevice);
}

void test()
{
  cuda::mr::cuda_async_memory_resource res{};

  { // allocate / deallocate
    auto* ptr = res.allocate(42);
    static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");
    ensure_device_ptr(ptr);

    res.deallocate(ptr, 42);
  }

  { // allocate / deallocate with alignment
    auto* ptr = res.allocate(42, 4);
    static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");
    ensure_device_ptr(ptr);

    res.deallocate(ptr, 42, 4);
  }

  { // allocate_async / deallocate_async
    cudaStream_t raw_stream;
    cudaStreamCreate(&raw_stream);
    cuda::stream_ref stream{raw_stream};

    auto* ptr = res.allocate_async(42, stream);
    static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");

    stream.wait();
    ensure_device_ptr(ptr);

    res.deallocate_async(ptr, 42, stream);
    cudaStreamDestroy(raw_stream);
  }

  { // allocate_async / deallocate_async with alignment
    cudaStream_t raw_stream;
    cudaStreamCreate(&raw_stream);
    cuda::stream_ref stream{raw_stream};

    auto* ptr = res.allocate_async(42, 4, stream);
    static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");

    stream.wait();
    ensure_device_ptr(ptr);

    res.deallocate_async(ptr, 42, 4, stream);
    cudaStreamDestroy(raw_stream);
  }

#ifndef TEST_HAS_NO_EXCEPTIONS
  { // allocate with too small alignment
    while (true)
    {
      try
      {
        auto* ptr = res.allocate(5, 42);
        unused(ptr);
      }
      catch (const cuda::std::bad_alloc&)
      {
        break;
      }
      assert(false);
    }
  }

  { // allocate with non matching alignment
    while (true)
    {
      try
      {
        auto* ptr = res.allocate(5, 1337);
        unused(ptr);
      }
      catch (const cuda::std::bad_alloc&)
      {
        break;
      }
      assert(false);
    }
  }
  { // allocate_async with too small alignment
    while (true)
    {
      cudaStream_t raw_stream;
      cudaStreamCreate(&raw_stream);
      try
      {
        auto* ptr = res.allocate_async(5, 42, raw_stream);
        unused(ptr);
      }
      catch (cuda::std::bad_alloc&)
      {
        cudaStreamDestroy(raw_stream);
        break;
      }
      assert(false);
    }
  }

  { // allocate_async with non matching alignment
    while (true)
    {
      cudaStream_t raw_stream;
      cudaStreamCreate(&raw_stream);
      try
      {
        auto* ptr = res.allocate_async(5, 1337, raw_stream);
        unused(ptr);
      }
      catch (cuda::std::bad_alloc&)
      {
        cudaStreamDestroy(raw_stream);
        break;
      }
      assert(false);
    }
  }
#endif // TEST_HAS_NO_EXCEPTIONS
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, test();)
  return 0;
}
