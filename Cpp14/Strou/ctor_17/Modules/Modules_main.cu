//------------------------------------------------------------------------------
/// \file Modules_main.cu
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Main driver function for demonstrating linear memory arrays as
/// modules, mathematically.
/// \ref Ch. 17 Constructors; Bjarne Stroustrup, 
///   The C++ Programming Language, 4th Ed., Stroustrup.
/// 3.2.2. Device Memory of 3.2 CUDA C Runtime of Programming Guide of
/// CUDA Toolkit Documentation
/// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-c-runtime  
/// \details RAII for CUDA C-style arrays or i.e. linear memory, with operator
/// overloading so linear memory are modules, mathematically.
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
///   nvcc -std=c++14 Modules_main.cu -o Modules_main
//------------------------------------------------------------------------------
#include "Modules.h"

#include <iostream>

using CUDA::Modules::TupleMalloc;

int main()
{
  {
    // TupleMallocDefaultConstructs
    TupleMalloc<float, 4, 1> test_tuple_malloc;
  }

  {
    // TupleMallocIncrementAdds
    // size or "length" L of the tuple or linear memory array.
    constexpr std::size_t L {128};
    // number of threads in a single thread block.
    constexpr std::size_t N_x {16};

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
      h_a[i] = 1.f * i;
      h_b[i] = 2.f * i;
    }

    // TupleMallocConstructsFromCArray
    TupleMalloc<float, L, N_x> d_a {h_a};
    TupleMalloc<float, L, N_x> d_b {h_b};
    {
      float* h_a_test;
      float* h_b_test;
      h_a_test = (float*)malloc(L * sizeof(float));
      h_b_test = (float*)malloc(L * sizeof(float));

      d_a.copy_to_host(h_a_test);
      d_b.copy_to_host(h_b_test);

      for (int i {0}; i < 5; ++i)
      {
        std::cout << h_a_test[i] << ' ' << h_b_test[i] << ' ';
        // 0, 1, 2, 3, 4...
      }

      free(h_a_test);
      free(h_b_test);
    }
    d_a += d_b;

    d_a.copy_to_host(h_c);

    for (int i {0}; i < 5; ++i)
    {
      std::cout << h_c[i] << ' '; // 0, 3, 6, 9, 12...
    }

    // TupleMallocBinaryAdds
    TupleMalloc<float, L, N_x> d_c;
    // {d_a + d_b};
    d_c = d_a + d_b;
    {
      float* h_c_test;
      h_c_test = (float*)malloc(L * sizeof(float));
      d_c.copy_to_host(h_c_test);
      std::cout << '\n';
      for (int i {0}; i < 5; ++i)
      {
        std::cout << h_c_test[i] << ' ';
        // 0, 5, 10, 15, 20...
      }
      std::cout << '\n';
      free(h_c_test);
    }

    // TupleMallocDecrementSubtracts.
    d_c -= d_b;
    {
      float* h_c_test;
      h_c_test = (float*)malloc(L * sizeof(float));
      d_c.copy_to_host(h_c_test);
      std::cout << '\n';
      for (int i {0}; i < 5; ++i)
      {
        std::cout << h_c_test[i] << ' ';
        // 0, 3, 6, 9, 12...
      }
      std::cout << '\n';
      free(h_c_test);
    }

    // TupleMallocBinarySubtracts
    {
      std::cout << "\n TupleMallocBinarySubtracts \n";

      TupleMalloc<float, L, N_x> d_test {d_b - d_c}; // binary subtraction.
      {
        float* h_test;
        h_test = (float*)malloc(L * sizeof(float));
        d_test.copy_to_host(h_test);
        std::cout << '\n';
        for (int i {0}; i < 5; ++i)
        {
          std::cout << h_test[i] << ' ';
          // 0, -1, -2, -3, -4...
        }
        std::cout << '\n';
        free(h_test);
      }
    }

    // TupleMallocScalarMultiplies
    {
      std::cout << "\n TupleMallocScalarMultiplies \n";

      d_c *= 0.1f;

      {
        float* h_test;
        h_test = (float*)malloc(L * sizeof(float));
        d_c.copy_to_host(h_test);
        std::cout << '\n';
        for (int i {0}; i < 5; ++i)
        {
          std::cout << h_test[i] << ' ';
          // 0, 0.3 0.6 0.9 1.2
        }
        std::cout << '\n';
        free(h_test);
      }
    }

    // TupleMallocScalarDivides
    {
      std::cout << "\n TupleMallocScalarDivides \n";

      d_c /= 2.0f;

      {
        float* h_test;
        h_test = (float*)malloc(L * sizeof(float));
        d_c.copy_to_host(h_test);
        std::cout << '\n';
        for (int i {0}; i < 5; ++i)
        {
          std::cout << h_test[i] << ' ';
          // 0, 0.3 0.6 0.9 1.2
        }
        std::cout << '\n';
        free(h_test);
      }
    }

    free(h_a);
    free(h_b);
    free(h_c);
  }
}
