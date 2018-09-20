//------------------------------------------------------------------------------
/// \file LinearMemory_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Main driver function for demonstrating linear memory arrays.
/// \ref Ch. 17 Constructors; Bjarne Stroustrup, 
///   The C++ Programming Language, 4th Ed., Stroustrup.
/// 3.2.2. Device Memory of 3.2 CUDA C Runtime of Programming Guide of
/// CUDA Toolkit Documentation
/// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-c-runtime  
/// \details RAII for CUDA C-style arrays or i.e. linear memory.
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
#include "LinearMemory.h"

#include <cstddef> // std::size_t
#include <cmath> // sin, cos
#include <iostream>

using CUDA::LinearMemory::Array;
using CUDA::LinearMemory::ArrayMalloc;

template <typename T>
__global__ void vector_addition(T*a, T* b, T* c, const std::size_t M)
{
  // Get our global thread ID.
  std::size_t tid {blockIdx.x * blockDim.x + threadIdx.x};

  // Make sure we don't go out of bounds.
  if (tid < M)
  {
    c[tid] = a[tid] + b[tid];
  }
}

int main()
{
  {
    // ArrayMallocAdds
    // \ref https://www.olcf.ornl.gov/tutorials/cuda-vector-addition/
    // Size of vectors
    constexpr std::size_t N {100000};

    // Host input vectors
    float *h_a;
    float *h_b;
    // Host output vector
    float *h_c;

    h_a = (float*)malloc(N * sizeof(float));
    h_b = (float*)malloc(N * sizeof(float));
    h_c = (float*)malloc(N * sizeof(float));

    // Initialize vectors on host
    for (std::size_t i {0}; i < N; i++)
    {
      h_a[i] = sin(i) * sin(i);
      h_b[i] = cos(i) * cos(i);
    }

    for (int i {0}; i < 5; i++)
    {
      std::cout << h_a[i] << ' ' << h_b[i] << ' ';
    }

    ArrayMalloc<float, N> d_a {h_a};
    ArrayMalloc<float, N> d_b;
    d_b.set(h_b);    
    ArrayMalloc<float, N> d_c;    

    // ArrayMallocCopies.
    {
      float* h_a_test;
      float* h_b_test;

      h_a_test = (float*)malloc(N * sizeof(float));
      h_b_test = (float*)malloc(N * sizeof(float));

      cudaMemcpy(h_a_test, d_a(), N * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_b_test, d_b.get(), N * sizeof(float), cudaMemcpyDeviceToHost);

//      d_a.get(h_a_test);

      for (int i {0}; i < 5; i++)
      {
        std::cout << h_a_test[i] << ' ' << h_b_test[i] << ' ';
      }

      free(h_a_test);
      free(h_b_test);
    }    

    // Number of threads in each thread block
    std::size_t blockSize {1024};

    // Number of thread blocks in grid.
    std::size_t gridSize = (float)ceil((float)N/blockSize);

    // Execute the addition kernel
    vector_addition<float><<<gridSize, blockSize>>>(d_a(), d_b(), d_c(), N);

    d_c.copy_to_h(h_c);

    // Sum up vector c and print result divided by N, this should equal 1.
    float sum {0};
    for (int i {0}; i < N; i++)
    {
      sum += h_c[i];
    }
    std::cout << "final result : " << sum << std::endl;

    free(h_a);
    free(h_b);
    free(h_c);
  }
}

