//------------------------------------------------------------------------------
/// \file LinearMemory.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Device memory allocated as linear memory.  
/// \ref 3.2.2. Device Memory of 3.2 CUDA C Runtime of Programming Guide of
/// CUDA Toolkit Documentation
/// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-c-runtime     
/// https://devtalk.nvidia.com/default/topic/938748/will-this-cause-troubles-/
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
///   nvcc -std=c++14 LinearMemory_main.cu -o LinearMemory_main
//------------------------------------------------------------------------------
#ifndef _CUDA_LINEAR_MEMORY_H_
#define _CUDA_LINEAR_MEMORY_H_

#include "../Utilities/Exception.h" // check_cuda_error

#include <cstddef> // std::size_t
#include <cuda_runtime_api.h> // cudaMallocManaged, cudaFree
#include <utility> // std::move, std::swap

#include <iostream>

namespace CUDA
{

namespace LinearMemory
{

//------------------------------------------------------------------------------
/// \brief Linear memory allocated using cudaMalloc
//------------------------------------------------------------------------------
template <typename T, std::size_t L>
class ArrayMalloc
{
  public:

    ArrayMalloc()
    {
      cudaError_t cuda_error {cudaMalloc((void**)&d_data_, size_)};
      CUDA::Utilities::Exceptions::check_cuda_error(cuda_error);

      cuda_error = cudaMemset(d_data_, 0, size_);
      CUDA::Utilities::Exceptions::check_cuda_error(cuda_error);
    }

    //--------------------------------------------------------------------------
    /// \brief Constructor for a public interface (API) from host C-style arrays
    //--------------------------------------------------------------------------
    explicit ArrayMalloc(T* h_data)
    {
      cudaError_t cuda_error {cudaMalloc((void**)&d_data_, size_)};
      CUDA::Utilities::Exceptions::check_cuda_error(cuda_error);

      cuda_error = cudaMemcpy(d_data_, h_data, size_, cudaMemcpyHostToDevice);
      CUDA::Utilities::Exceptions::check_cuda_error(cuda_error);
    }

    // Copyable.
    ArrayMalloc(const ArrayMalloc& a)               // copy constructor
    {
      cudaError_t cuda_error {cudaMalloc((void**)&d_data_, size_)};
      CUDA::Utilities::Exceptions::check_cuda_error(cuda_error);

      cuda_error =
        cudaMemcpy(d_data_, a.d_data_, size_, cudaMemcpyDeviceToDevice);
      CUDA::Utilities::Exceptions::check_cuda_error(cuda_error);      
    }

    ArrayMalloc& operator=(const ArrayMalloc& a)    // copy assignment
    {
      cudaError_t cuda_error {
        cudaMemcpy(d_data_, a.d_data_, size_, cudaMemcpyDeviceToDevice)
      };
      CUDA::Utilities::Exceptions::check_cuda_error(cuda_error);            
      return *this;
    }

    // Movable.
    /// \ref https://docs.microsoft.com/en-us/cpp/cpp/move-constructors-and-move-assignment-operators-cpp?view=vs-2017
    ArrayMalloc(ArrayMalloc&& a):
      d_data_{nullptr}
    {
      d_data_ = a.d_data_;
      a.d_data_ = nullptr;
    }    

    ArrayMalloc& operator=(ArrayMalloc&& a)
    {
      // Perform no operation if you try to assign the object to itself.
      if (this != &a)
      {
        cudaError_t cuda_error {cudaFree(d_data_)};
        CUDA::Utilities::Exceptions::check_cuda_error(cuda_error);

        //d_data_ = a.d_data_;
        std::swap(d_data_, a.d_data_);
        a.d_data_ = nullptr;        
      }
      return *this;
    }

    virtual ~ArrayMalloc()
    {
      if (d_data_ != nullptr)
      {
        cudaError_t cuda_error {cudaFree(d_data_)};
        CUDA::Utilities::Exceptions::check_cuda_error(cuda_error);
      }
    }

    //--------------------------------------------------------------------------
    /// \brief Setter interface (API) from host C-style arrays
    //--------------------------------------------------------------------------
    ArrayMalloc& operator()(T* h_data)
    {
      cudaError_t cuda_error {
        cudaMemcpy(d_data_, h_data, size_, cudaMemcpyHostToDevice)
      };
      CUDA::Utilities::Exceptions::check_cuda_error(cuda_error);

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
      CUDA::Utilities::Exceptions::check_cuda_error(cuda_error);
    }

    void copy_to_host(T* h_data) const
    {
      cudaError_t cuda_error {
        cudaMemcpy(h_data, d_data_, size_, cudaMemcpyDeviceToHost)};
      CUDA::Utilities::Exceptions::check_cuda_error(cuda_error);
    }


  protected:

    //--------------------------------------------------------------------------
    /// \brief Accessor to underlying CUDA C-style array
    //--------------------------------------------------------------------------
    T* data()
    {
      return d_data_;
    }

    //--------------------------------------------------------------------------
    /// \brief Getter interface (API) from linear memory to host C-style arrays
    //--------------------------------------------------------------------------
    // TO-DO: deprecated.
    // void operator()(T* h_data) const
    // {
    //  cudaError_t cuda_error {
    //    cudaMemcpy(h_data, d_data_, size_, cudaMemcpyDeviceToHost)};
    //  CUDA::Utilities::Exceptions::check_cuda_error(cuda_error);
    //}

  private:

    static constexpr std::size_t size_ {L * sizeof(T)};
    /// \ref https://eli.thegreenplace.net/2009/10/21/are-pointers-and-arrays-equivalent-in-c
    T* d_data_;
};

//------------------------------------------------------------------------------
/// \brief Linear memory allocated using cudaMallocManaged
//------------------------------------------------------------------------------
template <typename T, std::size_t L>
class Array
{
  public:

    Array()
    {
      cudaError_t cuda_error {cudaMalloc((void**)&d_data_, size_)};
      CUDA::Utilities::Exceptions::check_cuda_error(cuda_error);

      cuda_error = cudaMemset(d_data_, 0, size_);
      CUDA::Utilities::Exceptions::check_cuda_error(cuda_error);
    }

    //--------------------------------------------------------------------------
    /// \brief Constructor for a public interface (API) from host C-style arrays
    //--------------------------------------------------------------------------
    explicit Array(T* h_data)
    {
      cudaError_t cuda_error {cudaMalloc((void**)&d_data_, size_)};
      CUDA::Utilities::Exceptions::check_cuda_error(cuda_error);

      cuda_error = cudaMemcpy(d_data_, h_data, size_, cudaMemcpyHostToDevice);
      CUDA::Utilities::Exceptions::check_cuda_error(cuda_error);
    }

    // Copyable.
    Array(const Array& a)               // copy constructor
    {
      cudaError_t cuda_error {cudaMalloc((void**)&d_data_, size_)};
      CUDA::Utilities::Exceptions::check_cuda_error(cuda_error);

      cuda_error =
        cudaMemcpy(d_data_, a.d_data_, size_, cudaMemcpyDeviceToDevice);
      CUDA::Utilities::Exceptions::check_cuda_error(cuda_error);      
    }

    Array& operator=(const Array& a)    // copy assignment
    {
      cudaError_t cuda_error {
        cudaMemcpy(d_data_, a.d_data_, size_, cudaMemcpyDeviceToDevice)
      };
      CUDA::Utilities::Exceptions::check_cuda_error(cuda_error);            
      return *this;
    }

    // Movable.
    /// \ref https://docs.microsoft.com/en-us/cpp/cpp/move-constructors-and-move-assignment-operators-cpp?view=vs-2017
    Array(Array&& a):
      d_data_{nullptr}
    {
      d_data_ = a.d_data_;
      a.d_data_ = nullptr;
    }    

    Array& operator=(Array&& a)
    {
      // Perform no operation if you try to assign the object to itself.
      if (this != &a)
      {
        cudaError_t cuda_error {cudaFree(d_data_)};
        CUDA::Utilities::Exceptions::check_cuda_error(cuda_error);

        //d_data_ = a.d_data_;
        std::swap(d_data_, a.d_data_);
        a.d_data_ = nullptr;        
      }
      return *this;
    }

    virtual ~Array()
    {
      if (d_data_ != nullptr)
      {
        cudaError_t cuda_error {cudaFree(d_data_)};
        CUDA::Utilities::Exceptions::check_cuda_error(cuda_error);
      }
    }

    //--------------------------------------------------------------------------
    /// \brief Setter interface (API) from host C-style arrays
    //--------------------------------------------------------------------------
    void set(T* h_data)
    {
      cudaError_t cuda_error {
        cudaMemcpy(d_data_, h_data, size_, cudaMemcpyHostToDevice)
      };
      CUDA::Utilities::Exceptions::check_cuda_error(cuda_error);
    }

    //--------------------------------------------------------------------------
    /// \brief Getter interface (API) from host C-style arrays
    /// \details Copies from device to the host argument array.
    //--------------------------------------------------------------------------
    void get(T* h_data) const
    {
      cudaError_t cuda_error {
        cudaMemcpy(h_data, d_data_, size_, cudaMemcpyDeviceToHost)};
      CUDA::Utilities::Exceptions::check_cuda_error(cuda_error);
    }

  protected:

    //--------------------------------------------------------------------------
    /// \brief Accessor to underlying CUDA C-style array
    //--------------------------------------------------------------------------
    T* data()
    {
      return d_data_;
    }

  private:

    static constexpr std::size_t size_ {L * sizeof(T)};
    T* d_data_;
};

//------------------------------------------------------------------------------
/// \brief Linear memory allocated using cudaMallocPitch
//------------------------------------------------------------------------------
template <typename T, std::size_t L_1, std::size_t L_2>
class ArrayPitched
{
  public:

    ArrayPitched()
    {
      cudaMallocPitch((void**)&d_data_, &pitch_, L_1 * sizeof(T), L_2);
    }

    ~ArrayPitched()
    {
      cudaFree(d_data_);
    }

  private:

    T d_data_[L_1 * L_2];
    std::size_t pitch_;
};

//------------------------------------------------------------------------------
/// \brief Linear memory allocated using cudaMalloc3D
//------------------------------------------------------------------------------
template <typename T, std::size_t L_1, std::size_t L_2, std::size_t L_3>
class Array3D
{
  public:

    Array3D():
    // error: there are no arguments to ‘make_cudaExtent’ that depend on a
    // template parameter, so a declaration of ‘make_cudaExtent’ must be
    // available
    //  extent_{make_cudaExtent(L_1 * sizeof(T), L_2, L_3)}
      extent_{L_3, L_2, L_1}
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


