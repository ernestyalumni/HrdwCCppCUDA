//------------------------------------------------------------------------------
/// \file Modules.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Device memory allocated as linear memory as fields.  
/// \ref 3.2.2. Device Memory of 3.2 CUDA C Runtime of Programming Guide of
/// CUDA Toolkit Documentation
/// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-c-runtime     
/// \details RAII for CUDA C-style arrays or linear memory, mathematically as
/// fields.
/// \copyright If you find this code useful, feel free to donate directly and
/// easily at this direct PayPal link: 
///
/// https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
/// 
/// which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.
/// Otherwise, I receive emails and messages on how all my (free) material on
/// physics, math, and engineering have helped students with their studies, and
/// I know what it's like to not have money as a student, but love physics (or 
/// math, sciences, etc.), so I am committed to keeping all my material 
/// open-source and free, whether or not sufficiently crowdfunded, under the 
/// open-source MIT license: feel free to copy, edit, paste, make your own
/// versions, share, use as you wish.    
/// Peace out, never give up! -EY
//------------------------------------------------------------------------------
/// COMPILATION TIPS:
///  nvcc -std=c++14 Array_main.cpp -o Array_main
//------------------------------------------------------------------------------
#ifndef _CUDA_MODULES_MODULES_H_
#define _CUDA_MODULES_MODULES_H_

#include "../LinearMemory/LinearMemory.h"

#include <cmath> // ceil
#include <stdexcept> // std::runtime_error
#include <utility> // std::move, std::swap

namespace CUDA
{

namespace Modules
{

// Device unary operators

template <typename T>
__global__ void tuple_1d_increment(T* a, T* b, const std::size_t L)
{
  // Get our global thread ID.
  std::size_t k_x {blockIdx.x * blockDim.x + threadIdx.x};

  // Make sure we don't go out of bounds.
  if (k_x >= L) { return; }  

  for (std::size_t tid {k_x}; tid < L; tid += gridDim.x * blockDim.x)
  {
    a[tid] += b[tid];
  }
}

template <typename T>
__global__ void tuple_1d_decrement(T* a, T* b, const std::size_t L)
{
  // Get our global thread ID.
  std::size_t k_x {blockIdx.x * blockDim.x + threadIdx.x};

  // Make sure we don't go out of bounds.
  if (k_x >= L) { return; }  

  for (std::size_t tid {k_x}; tid < L; tid += gridDim.x * blockDim.x)
  {
    a[tid] -= b[tid];
  }
}

template <typename T>
__global__ void tuple_1d_scalar_multiplication(
  T* a,
  const T c,
  const std::size_t L)
{
  // Get our global thread ID.
  std::size_t k_x {blockIdx.x * blockDim.x + threadIdx.x};

  // Make sure we don't go out of bounds.
  if (k_x >= L) { return; }  

  for (std::size_t tid {k_x}; tid < L; tid += gridDim.x * blockDim.x)
  {
    a[tid] *= c;
  }
}


// Device binary operators

template <typename T>
__global__ void tuple_1d_addition(T* a, T* b, T* c, const std::size_t L)
{
  // Get our global thread ID.
  std::size_t k_x {blockIdx.x * blockDim.x + threadIdx.x};

  // Make sure we don't go out of bounds.
  if (k_x >= L) { return; }  

  for (std::size_t tid {k_x}; tid < L; tid += gridDim.x * blockDim.x)
  {
    c[tid] = a[tid] + b[tid];
  }
}

template <typename T, std::size_t L, std::size_t N_x>
class TupleMalloc : CUDA::LinearMemory::ArrayMalloc<T, L>
{
  public:

    using ArrayMalloc = CUDA::LinearMemory::ArrayMalloc<T, L>;
//    using ArrayMalloc::ArrayMalloc;
    using ArrayMalloc::copy_to_host;

    TupleMalloc() = default;

    //--------------------------------------------------------------------------
    /// \brief Constructor for a public interface (API) from host C-style arrays
    //--------------------------------------------------------------------------
    explicit TupleMalloc(T* h_data):
      ArrayMalloc{h_data}
    {}

    // unary arithmetic
    TupleMalloc& operator+=(TupleMalloc& b)
    {
      tuple_1d_increment<T><<<M_x, N_x>>>(this->data(), b.data(), L);
      return *this;      
    }

    TupleMalloc& operator-=(TupleMalloc& b)
    {
      tuple_1d_decrement<T><<<M_x, N_x>>>(this->data(), b.data(), L);
      return *this;      
    }

    TupleMalloc& operator*=(const T c)
    {
      tuple_1d_scalar_multiplication<T><<<M_x, N_x>>>(this->data(), c, L);
      return *this;      
    }

    TupleMalloc& operator/=(const T c)
    {
      if (c == 0)
      {
        throw std::runtime_error("Division by 0 error");
      }
      tuple_1d_scalar_multiplication<T><<<M_x, N_x>>>(this->data(), 1/c, L);
      return *this;      
    }

    // binary arithmetic
    template <typename U, std::size_t P, std::size_t Q_x>
    friend TupleMalloc<U, P, Q_x>& operator+(TupleMalloc<U, P, Q_x>& a,
      TupleMalloc<U, P, Q_x>& b);

    template <typename U, std::size_t P, std::size_t Q_x>
    friend TupleMalloc<U, P, Q_x>& operator-(TupleMalloc<U, P, Q_x>& a,
      TupleMalloc<U, P, Q_x>& b);

  protected:

    using ArrayMalloc::data;

  private:

    // Number of thread blocks in a grid.
    static constexpr std::size_t M_x {(L + N_x -1)/N_x};
};

// binary arithmetic
template<typename T, std::size_t L, std::size_t N_x>
TupleMalloc<T, L, N_x>& operator+(TupleMalloc<T, L, N_x>& a,
  TupleMalloc<T, L, N_x>& b)
{
  return a += b;
}

template<typename T, std::size_t L, std::size_t N_x>
TupleMalloc<T, L, N_x>& operator-(TupleMalloc<T, L, N_x>& a,
  TupleMalloc<T, L, N_x>& b)
{
  return a -= b;
}

} // namespace Modules

} // namespace CUDA

#endif // _CUDA_MODULES_MODULES_H_
