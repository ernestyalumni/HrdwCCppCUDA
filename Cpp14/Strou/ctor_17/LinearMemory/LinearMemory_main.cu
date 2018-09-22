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
///   nvcc -std=c++14 LinearMemory_main.cu -o LinearMemory_main
//------------------------------------------------------------------------------
#include "LinearMemory.h"

#include <cstddef> // std::size_t
#include <cmath> // sin, cos
#include <iostream>
#include <utility> // std::move

using CUDA::LinearMemory::Array;
using CUDA::LinearMemory::ArrayMalloc;

template <typename T, std::size_t L>
class TestArrayMalloc : public ArrayMalloc<T, L>
{
  public:

    using ArrayMalloc = ArrayMalloc<T, L>;
    using ArrayMalloc::ArrayMalloc;

    using ArrayMalloc::data;
};

template <typename T, std::size_t L>
class TestArray : public Array<T, L>
{
  public:

    using Array = Array<T, L>;
    using Array::Array;

    using Array::data;
};


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

    TestArrayMalloc<float, N> d_a {h_a};
    TestArrayMalloc<float, N> d_b;
    d_b.set(h_b);    
    // this works as well: d_b(h_b);
    TestArrayMalloc<float, N> d_c;    

    // ArrayMallocCopies.
    {
      float* h_a_test;
      float* h_b_test;

      h_a_test = (float*)malloc(N * sizeof(float));
      h_b_test = (float*)malloc(N * sizeof(float));

      cudaMemcpy(h_a_test, d_a.data(), N * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_b_test, d_b.data(), N * sizeof(float), cudaMemcpyDeviceToHost);

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
    vector_addition<float><<<gridSize, blockSize>>>(
      d_a.data(),
      d_b.data(),
      d_c.data(), N);

    //d_c(h_c);
    d_c.copy_to_host(h_c);

    for (int i {0}; i < 5; i++)
    {
      std::cout << h_c[i] << ' ';
    }


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

  // ArrayFollowsRuleOf5
  {
    constexpr std::size_t L {100000};

    // Host input vectors
    float *h_a;
    float *h_b;
    // Host output vector
    float *h_c;

    h_a = (float*)malloc(L * sizeof(float));
    h_b = (float*)malloc(L * sizeof(float));
    h_c = (float*)malloc(L * sizeof(float));

    // Initialize vectors on host
    for (std::size_t i {0}; i < L; i++)
    {
      h_a[i] = sin(i) * sin(i);
      h_b[i] = cos(i) * cos(i);
    }

    // ArrayDefaultConstructsTo0
    std::cout << "\n ArrayDefaultConstructsTo0 \n";
    Array<float, L> d_a;

    {
      float* h_a_test;
      h_a_test = (float*)malloc(L * sizeof(float));

      d_a.get(h_a_test);

      for (int i {0}; i < 5; i++)
      {
        std::cout << h_a_test[i] << ' '; // should be all 0
      }
      std::cout << std::endl;
      for (int i {L-5}; i < L; i++)
      {
        std::cout << h_a_test[i] << ' '; // should be all 0
      }

      free(h_a_test);
    }    

    // ArraySetsAndGetsFromCArray
    std::cout << "\n ArraySetsAndGetsFromCArray \n";
    {
      float* h_a_test;

      h_a_test = (float*)malloc(L * sizeof(float));
      d_a.set(h_a);
      d_a.get(h_a_test);

      for (int i {0}; i < 5; i++)
      {
        std::cout << h_a_test[i] << ' ' << h_a[i] << ' '; // should be the same
      }
      std::cout << std::endl;
      for (int i {L-5}; i < L; i++)
      {
        std::cout << h_a_test[i] << ' ' << h_a[i] << ' '; // should be the same
      }
      std::cout << std::endl;

      free(h_a_test);
    }    

    // ArrayCopies.
    std::cout << "\n ArrayCopies \n";
    {
      // ArrayCopyConstructs.
      Array<float, L> d_a_copy {d_a}; // copy construct.

      float* h_a_test;
      h_a_test = (float*)malloc(L * sizeof(float));
      d_a_copy.get(h_a_test);
      for (int i {0}; i < 5; i++)
      {
        std::cout << h_a_test[i] << ' '; // should be sin^2
      }
      std::cout << std::endl;
      for (int i {L-5}; i < L; i++)
      {
        std::cout << h_a_test[i] << ' '; // should be sin^2
      }

      // Check independence of d_a and d_a_copy
      float* h_a_test2;
      h_a_test2 = (float*)malloc(L * sizeof(float));
      d_a.get(h_a_test2);
      for (int i {0}; i < 5; i++)
      {
        std::cout << h_a_test2[i] << ' '; // should be sin^2
      }
      std::cout << std::endl;
      for (int i {L-5}; i < L; i++)
      {
        std::cout << h_a_test2[i] << ' '; // should be sin^2
      }

      free(h_a_test);
      free(h_a_test2);
    }    

    {
      // ArrayCopyAssigns.
      const Array<float, L> d_a_test {d_a};

      Array<float, L> d_a_copy;
      d_a_copy = d_a_test; // copy assign.

      float* h_a_test;
      h_a_test = (float*)malloc(L * sizeof(float));
      d_a_copy.get(h_a_test);
      for (int i {0}; i < 5; i++)
      {
        std::cout << h_a_test[i] << ' '; // should be sin^2
      }
      std::cout << std::endl;
      for (int i {L-5}; i < L; i++)
      {
        std::cout << h_a_test[i] << ' '; // should be sin^2
      }

      // Check independence of d_a and d_a_copy
      float* h_a_test2;
      h_a_test2 = (float*)malloc(L * sizeof(float));
      d_a_test.get(h_a_test2);
      for (int i {0}; i < 5; i++)
      {
        std::cout << h_a_test2[i] << ' '; // should be sin^2
      }
      std::cout << std::endl;
      for (int i {L-5}; i < L; i++)
      {
        std::cout << h_a_test2[i] << ' '; // should be sin^2
      }

      free(h_a_test);
      free(h_a_test2);
    }    

    // ArrayMoves.
    std::cout << "\n ArrayMoves \n";
    {
      // ArrayMoveConstructs.
      TestArray<float, L> d_b_copy {h_b};
      std::cout << "\nMove Constructing d_b_moved\n";
      TestArray<float, L> d_b_moved {std::move(d_b_copy)};

      float* h_b_test;
      h_b_test = (float*)malloc(L * sizeof(float));
      d_b_moved.get(h_b_test);
      for (int i {0}; i < 5; i++)
      {
        std::cout << h_b_test[i] << ' '; // should be cos^2
      }
      std::cout << std::endl;
      for (int i {L-5}; i < L; i++)
      {
        std::cout << h_b_test[i] << ' '; // should be cos^2
      }

      // MovedArraySetTonullptr
      std::cout << std::boolalpha << (d_b_copy.data() == nullptr) << '\n'; 

      free(h_b_test);
    }

    {
      // ArrayMoveAssigns.
      TestArray<float, L> d_b_copy {h_b};
      TestArray<float, L> d_b_moved; 
      d_b_moved = std::move(d_b_copy);

      float* h_b_test;
      h_b_test = (float*)malloc(L * sizeof(float));
      d_b_moved.get(h_b_test);
      for (int i {0}; i < 5; i++)
      {
        std::cout << h_b_test[i] << ' '; // should be cos^2
      }
      std::cout << std::endl;
      for (int i {L-5}; i < L; i++)
      {
        std::cout << h_b_test[i] << ' '; // should be cos^2
      }

      // MovedArraySetTonullptr
      std::cout << std::boolalpha << (d_b_copy.data() == nullptr) << '\n'; 

      free(h_b_test);
    }

    free(h_a);
    free(h_b);
    free(h_c);
  } // ArrayFollowsRuleOf5

}

