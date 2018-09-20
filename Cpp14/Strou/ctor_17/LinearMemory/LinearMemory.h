//------------------------------------------------------------------------------
/// \file LinearMemory.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Device memory allocated as linear memory.  
/// \ref 3.2.2. Device Memory of 3.2 CUDA C Runtime of Programming Guide of
/// CUDA Toolkit Documentation
/// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-c-runtime     
/// \details RAII for CUDA C-style arrays or linear memory.
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
#ifndef _CUDA_LINEAR_MEMORY_H_
#define _CUDA_LINEAR_MEMORY_H_

#include <cstddef> // std::size_t
#include <cuda_runtime_api.h> // cudaMallocManaged, cudaFree

#include <iostream>

namespace CUDA
{

namespace LinearMemory
{

//------------------------------------------------------------------------------
/// \brief Linear memory allocated using cudaMalloc
//------------------------------------------------------------------------------
template <typename T, std::size_t N>
class ArrayMalloc
{
  public:

    ArrayMalloc()
    {
      cudaMalloc((void**)&d_data_, size_);
    }

    //--------------------------------------------------------------------------
    /// \brief Constructor for a public interface (API) from host C-style arrays
    //--------------------------------------------------------------------------
    explicit ArrayMalloc(T* h_data)
    {
      cudaMalloc((void**)&d_data_, size_);

      cudaError_t cuda_error {
        cudaMemcpy(d_data_, h_data, size_, cudaMemcpyHostToDevice)
      };

      if (cuda_error != cudaSuccess)
      {
        std::cout << " no cudaSuccess for cudaMemcpy upon ctor" << std::endl;
      }
    }

    ~ArrayMalloc()
    {
      cudaFree(d_data_);
    }

    //--------------------------------------------------------------------------
    /// \brief Setter interface (API) from host C-style arrays
    //--------------------------------------------------------------------------
    ArrayMalloc& operator()(T* h_data)
    {
      cudaError_t cuda_error {
        cudaMemcpy(d_data_, h_data, size_, cudaMemcpyHostToDevice)
      };

      if (cuda_error != cudaSuccess)
      {
        std::cout << " no cudaSuccess for cudaMemcpy " << std::endl;
      }

      return *this;
    }

    //--------------------------------------------------------------------------
    /// \brief Setter interface (API) from host C-style arrays
    //--------------------------------------------------------------------------
    void set(T* h_data)
    {
      cudaError_t cuda_error {
        cudaMemcpy(d_data_, h_data, size_, cudaMemcpyHostToDevice)
      };

      if (cuda_error != cudaSuccess)
      {
        std::cout << " no cudaSuccess for cudaMemcpy " << std::endl;
      }
    }


    // binary arithmetic
/*    template<typename U, std::size_t M>
    friend ArrayMalloc<U, M> operator+(
      ArrayMalloc<U, M> a, const ArrayMalloc<U, M>& b)
    {

    }
*/

    //--------------------------------------------------------------------------
    /// \brief Accessor to underlying CUDA C-style array
    //--------------------------------------------------------------------------
    T* operator()()
    {
      return d_data_;
    }

    T* get()
    {
      return d_data_;
    }

    //--------------------------------------------------------------------------
    /// \brief Getter interface (API) from host C-style arrays
    //--------------------------------------------------------------------------
    void copy_to_h(T* h_data) const
    {
      cudaMemcpy(h_data, d_data_, size_, cudaMemcpyDeviceToHost);
    }


  private:

    static constexpr std::size_t size_ {N * sizeof(T)};
    // \ref https://eli.thegreenplace.net/2009/10/21/are-pointers-and-arrays-equivalent-in-c
    T* d_data_;
};

//------------------------------------------------------------------------------
/// \brief Linear memory allocated using cudaMallocManaged
//------------------------------------------------------------------------------
template <typename T, std::size_t N>
class Array
{
  public:

    Array()
    {
      cudaMallocManaged((void**)&d_data_, size_);
    }

    ~Array()
    {
      cudaFree(d_data_);
    }

    //--------------------------------------------------------------------------
    /// \brief Public interface (API) from host C-style arrays
    //--------------------------------------------------------------------------
    Array& operator()(T* h_data)
    {
      cudaMemcpy(d_data_, h_data, size_, cudaMemcpyHostToDevice);
      return *this;
    }

  private:

    static constexpr std::size_t size_ {N * sizeof(T)};
    T d_data_[N];
};

//------------------------------------------------------------------------------
/// \brief Linear memory allocated using cudaMallocPitch
//------------------------------------------------------------------------------
template <typename T, std::size_t N_1, std::size_t N_2>
class ArrayPitched
{
  public:

    ArrayPitched()
    {
      cudaMallocPitch((void**)&d_data_, &pitch_, N_1 * sizeof(T), N_2);
    }

    ~ArrayPitched()
    {
      cudaFree(d_data_);
    }

  private:

    T d_data_[N_1 * N_2];
    std::size_t pitch_;
};

//------------------------------------------------------------------------------
/// \brief Linear memory allocated using cudaMalloc3D
//------------------------------------------------------------------------------
template <typename T, std::size_t N_1, std::size_t N_2, std::size_t N_3>
class Array3D
{
  public:

    Array3D():
    // error: there are no arguments to ‘make_cudaExtent’ that depend on a
    // template parameter, so a declaration of ‘make_cudaExtent’ must be
    // available
    //  extent_{make_cudaExtent(N_1 * sizeof(T), N_2, N_3)}
      extent_{N_3, N_2, N_1}
    {
      cudaMalloc3D(&dev_pitched_ptr_, extent_);
    }

    ~Array3D()
    {
      cudaFree(&dev_pitched_ptr_);
    }

  private:

    // \details cudaExtent struct has public variables 
    // size_t depth
    // size_t height
    // size_t width

    cudaExtent extent_;
    cudaPitchedPtr dev_pitched_ptr_;
};



} // namespace LinearMemory

} // namespace CUDA

#endif // _CUDA_ARRAY_H_


