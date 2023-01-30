#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace GoogleUnitTests
{
namespace CUDAToolkit
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(MemoryManagementTests, cudaMallocManagedAllocates)
{
  constexpr int N {1 << 20};
  float* x {nullptr};

  // Allocate Unified Memory -- accessible from CPU.
  cudaMallocManaged(&x, N * sizeof(float));

  cudaFree(x);

  SUCCEED();
}

} // namespace CUDAToolkit
} // namespace GoogleUnitTests
