#ifndef CUDA_SAMPLES_INTRODUCTION_VECTOR_ADDITION_CUH
#define CUDA_SAMPLES_INTRODUCTION_VECTOR_ADDITION_CUH

// For the CUDA runtime routines (prefixed with "cuda_").
#include <cuda_runtime.h>

namespace CUDASamples
{
namespace Introduction
{

__global__ void vector_add(
  const float* A,
  const float* B,
  float* C,
  std::size_t number_of_elements);

} // namespace Introduction
} // namespace CUDASamples

#endif // CUDA_SAMPLES_INTRODUCTION_VECTOR_ADDITION_CUH