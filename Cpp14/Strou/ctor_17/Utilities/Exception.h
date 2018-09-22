//------------------------------------------------------------------------------
/// \file Exceptions.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Error Handling functions.
/// \ref 5.3. Error Handling, 5. Modules, CUDA Runtime API, CUDA Toolkit Doc. 
/// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html
/// \details Exception helper functions.
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
#ifndef _CUDA_UTILITIES_EXCEPTIONS_H_
#define _CUDA_UTILITIES_EXCEPTIONS_H_

#include <stdexcept> // std::runtime_error
#include <string>

namespace CUDA
{

namespace Utilities
{

namespace Exceptions
{

inline void check_cuda_error(const cudaError_t cuda_error)
{
  if (cuda_error != cudaSuccess)
  {
    throw std::runtime_error(std::string{cudaGetErrorName(cuda_error)} + " " +
      std::string{cudaGetErrorString(cuda_error)});
  }
  return;
}

} // namespace Exceptions

} // namespace Utilities

} // namespace CUDA

#endif // _CUDA_UTILITIES_EXCEPTIONS_H_
