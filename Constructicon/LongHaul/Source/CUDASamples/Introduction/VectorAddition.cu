#include "VectorAddition.cuh"

#include <cuda_runtime.h>

namespace CUDASamples
{
namespace Introduction
{

__global__ void vector_add(
  const float* A,
  const float* B,
  float* C,
  std::size_t number_of_elements)
{
  std::size_t i {blockDim.x * blockIdx.x + threadIdx.x};

  if (i < number_of_elements)
  {
    C[i] = A[i] + B[i] + 0.0f;
  }
}

} // namespace Introduction
} // namespace CUDASamples