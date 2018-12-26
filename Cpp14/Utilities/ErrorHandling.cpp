//------------------------------------------------------------------------------
/// \file ErrorHandling.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief Source file for error handling C++ functors to check POSIX Linux
/// system call results.
/// \ref 
/// \details 
/// \copyright If you find this code useful, feel free to donate directly
/// (username ernestyalumni or email address above), going directly to: 
///
/// paypal.me/ernestyalumni
/// 
/// which won't go through a 3rd. party like indiegogo, kickstarter, patreon.
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
///  g++ -std=c++14 CheckReturns_main.cpp -o CheckReturns_main
//------------------------------------------------------------------------------
#include "ErrorHandling.h"

#include "Errno.h" // ErrorNumber

#include <string>
#include <system_error> // std::system_error, std::system_category

namespace Utilities
{
namespace ErrorHandling
{

int HandleReturnValue::operator()(
  int result,
  const std::string& custom_error_string)
{
  if (result < 0)
  {
    get_error_number();

    throw std::system_error(
      errno,
      std::system_category(),
      "Failed to " + custom_error_string + " with errno : " +
        error_number_.as_string() + "\n");

    return errno;
  }
  else
  {
    return result;
  }
}

int HandleReturnValue::operator()(int result)
{
  return this->operator()(
    result,
    "Integer return value to check was less than 0, and so,");
}

} // namespace ErrorHandling
} // namespace Utilities