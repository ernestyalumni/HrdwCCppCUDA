//------------------------------------------------------------------------------
/// \file Errno.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Wrappers and examples classes for errno and error handling.
/// \ref https://en.cppreference.com/w/cpp/error/errc
/// https://en.cppreference.com/w/cpp/error/errno_macros
/// \details Scoped enumeration (enum class) std::errc defines values of
/// portable error conditions corresponding to POSIX error codes.
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
///   g++ -std=c++17 Errno.cpp Errno_main.cpp -o Errno_main
//------------------------------------------------------------------------------
#include "Errno.h"

#include <system_error> // std::errc, std::error_category, std::system_category

namespace Utilities
{
namespace ErrorHandling
{

ErrorNumber::ErrorNumber() :
  error_number_{errno},
  error_code_{errno, std::system_category()},
  error_condition_{static_cast<std::errc>(errno)}
{}

ErrorNumber::ErrorNumber(const int error_number) :
  error_number_{error_number},
  error_code_{error_number, std::system_category()},
  error_condition_{static_cast<std::errc>(error_number)}
{}

ErrorNumber::ErrorNumber(const std::error_code& error_code) :
  error_number_{error_code.value()},
  error_code_{error_code},
  error_condition_{error_code.value(), error_code.category()}
{}

ErrorNumber::ErrorNumber(
  const int error_number,
  const std::error_category& ecat
  ) :
  error_number_{error_number},
  error_code_{error_number, ecat},
  error_condition_{error_number, ecat}
{}

} // namespace ErrorHandling
} // namespace Utilities
