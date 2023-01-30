#include <cstddef>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
// See https://en.cppreference.com/w/c/memory/malloc
#include <stdlib.h> // malloc

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

  EXPECT_NEQ(h_A, nullptr);
  EXPECT_NEQ(h_B, nullptr);
  EXPECT_NEQ(h_C, nullptr);

  // Free host memory.
  free(h_A);
  free(h_B);
  free(h_C);

  SUCCEED();
}

} // namespace CUDAIntroduction
} // namespace GoogleUnitTests
