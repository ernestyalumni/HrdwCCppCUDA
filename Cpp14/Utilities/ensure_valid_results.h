//------------------------------------------------------------------------------
/// \file ensure_valid_results.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Helper functions to check file descriptors.
/// \ref https://linux.die.net/man/3/clock_gettime     
/// \details 
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
///  g++ -std=c++14 Clocks_main.cpp -o Clocks_main
//------------------------------------------------------------------------------
#ifndef _UTILITIES_ENSURE_VALID_RESULTS_H_
#define _UTILITIES_ENSURE_VALID_RESULTS_H_

#include <cstring> // strerror
#include <iostream>
#include <string>
#include <system_error> 

namespace Utilities
{

//------------------------------------------------------------------------------
// \details Specified by Linux manual page, in that system calls that include
// - ::clock_gettime
// - ::clock_settime
// - ::clock_getres
// are documented to return 0 for success, or -1 for failure (in which case
// errno is set appropriately), and that errno will be returned by this
// function.
//------------------------------------------------------------------------------
int check_valid_fd(int e, const std::string& custom_error_string)
{
  if (e < 0)
  {
    std::cout << " errno : " << std::strerror(errno) << '\n';
    throw std::system_error(
      errno,
      std::generic_category(),
      "Failed to " + custom_error_string + "\n");
  }
  return errno;
}

} // namespace Utilities

#endif // _UTILITIES_CHECK_FDS_H_