//------------------------------------------------------------------------------
/// \file ErrorNumber.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Wrappers and examples classes for errno and error handling.
/// \ref https://en.cppreference.com/w/cpp/error/errc
/// https://en.cppreference.com/w/cpp/error/errno_macros
/// \details Scoped enumeration (enum class) std::errc defines values of
/// portable error conditions corresponding to POSIX error codes.
//------------------------------------------------------------------------------
#include "ErrorNumber.h"

#include <system_error> // std::errc, std::error_category, std::system_category

namespace Utilities
{
namespace ErrorHandling
{

ErrorNumber::ErrorNumber():
  error_number_{errno},
  error_code_{errno, std::system_category()},
  error_condition_{static_cast<std::errc>(errno)}
{}

ErrorNumber::ErrorNumber(const int error_number):
  error_number_{error_number},
  error_code_{error_number, std::system_category()},
  error_condition_{static_cast<std::errc>(error_number)}
{}

ErrorNumber::ErrorNumber(const std::error_code& error_code):
  error_number_{error_code.value()},
  error_code_{error_code},
  error_condition_{error_code.value(), error_code.category()}
{}

ErrorNumber::ErrorNumber(
  const int error_number,
  const std::error_category& ecat
  ):
  error_number_{error_number},
  error_code_{error_number, ecat},
  error_condition_{error_number, ecat}
{}

} // namespace ErrorHandling
} // namespace Utilities
