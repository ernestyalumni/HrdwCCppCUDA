#include <cstddef>
#include <cstdlib> // RAND_MAX
#include <cuda_runtime.h>
#include <gtest/gtest.h>
// See https://en.cppreference.com/w/c/memory/malloc
#include <stdlib.h> // malloc, rand()

namespace GoogleUnitTests
{
namespace CUDAIntroduction
{

// See https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/vectorAdd/vectorAdd.cu

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(VectorAddTests, AllocatesOnHostAndGPU)
{
  constexpr std::size_t number_of_elements {50000};
  constexpr std::size_t size {number_of_elements * sizeof(float)};

  // Allocate the host input vectors.

  float *h_A {static_cast<float*>(malloc(size))};
  float *h_B {static_cast<float*>(malloc(size))};
  float *h_C {static_cast<float*>(malloc(size))};

  EXPECT_NE(h_A, nullptr);
  EXPECT_NE(h_B, nullptr);
  EXPECT_NE(h_C, nullptr);

  for (std::size_t i {0}; i < number_of_elements; ++i)
  {
    // https://en.cppreference.com/w/cpp/numeric/random/RAND_MAX
    // RAND_MAX is implementation defined.
    h_A[i] = rand() / static_cast<float>(RAND_MAX);
    h_B[i] = rand() / static_cast<float>(RAND_MAX);
  }

  // Allocate device input vectors.

  float* d_A {nullptr};

  // See https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
  // __hist__device__cudaError_t cudaMalloc(void** devPtr, size_t size)
  cudaError_t err {cudaMalloc(reinterpret_cast<void**>(&d_A), size)};
  EXPECT_EQ(err, cudaSuccess);

  float* d_B {nullptr};
  err = cudaMalloc(reinterpret_cast<void**>(&d_B), size);
  EXPECT_EQ(err, cudaSuccess);

  float* d_C {nullptr};
  err = cudaMalloc(reinterpret_cast<void**>(&d_C), size);
  EXPECT_EQ(err, cudaSuccess);

  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  EXPECT_EQ(err, cudaSuccess);

  err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
  EXPECT_EQ(err, cudaSuccess);



  err = cudaFree(d_A);
  EXPECT_EQ(err, cudaSuccess);

  err = cudaFree(d_B);
  EXPECT_EQ(err, cudaSuccess);

  // Free host memory.
  free(h_A);
  free(h_B);
  free(h_C);

  SUCCEED();
}

} // namespace CUDAIntroduction
} // namespace GoogleUnitTests
