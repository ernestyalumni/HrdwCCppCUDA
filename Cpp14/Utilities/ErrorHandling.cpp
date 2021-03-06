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
///  g++ -std=c++17 Errno.cpp ErrorHandling.cpp ErrorHandling_main.cpp -o \
///   ErrorHandling_main
//------------------------------------------------------------------------------
#include "ErrorHandling.h"

#include "ErrorNumber.h" // ErrorNumber

#include <iostream>
#include <optional>
#include <string>
#include <system_error> // std::system_error, std::system_category

namespace Utilities
{
namespace ErrorHandling
{

HandleReturnValue::HandleReturnValue() :
  error_number_{}
{}

HandleReturnValue::HandleReturnValue(const int error_number) :
  error_number_{error_number}
{}

void HandleReturnValue::operator()(
  const int result,
  const std::string& custom_error_string)
{
  if (result < 0)
  {
    get_error_number();

    throw std::system_error(
      errno,
      std::system_category(),
      "Failed to " + custom_error_string + " with errno : " +
        error_number_.as_string() + " and error number " +
          std::to_string(error_number().error_number()) + '\n');
  }
}

void HandleReturnValue::operator()(const int result)
{
  this->operator()(
    result,
    "Integer return value to check was less than 0, and so,");
}

void HandleReturnValue::get_error_number()
{
  error_number_ = ErrorNumber{};
}

HandleReturnValuePassively::HandleReturnValuePassively():
  error_number_{}
{}

std::optional<ErrorNumber> HandleReturnValuePassively::optional()(
  const int return_value)
{
  if (return_value < 0)
  {
    get_error_number();

    return std::make_optional<ErrorNumber>(error_number_);
  }
  else
  {
    return std::nullopt;
  }
}

void HandleReturnValuePassively::get_error_number()
{
  error_number_ = ErrorNumber{};
}

std::optional<ErrorNumber> HandleClose::operator()(const int return_value)
{
  if (return_value < 0)
  {
    get_error_number();
    
    std::cerr << "Failed to close fd (::close()) with errno: " <<
      error_number_.as_string() << " and error number " <<
      std::to_string(error_number_.error_number()) << "\n";

    return std::make_optional<ErrorNumber>(error_number_);
  }

  return std::nullopt;
}

void HandleRead::operator()(const ssize_t number_of_bytes)
{
  if (number_of_bytes < 0)
  {
    get_error_number();

    throw std::system_error(
      errno,
      std::system_category(),
      "Failed to ::read from fd with errno : " + error_number().as_string() +
        '\n');
  }
  else if (number_of_bytes == 0)
  {
    std::cout << "End of file reached for fd\n";
  }
}


} // namespace ErrorHandling
} // namespace Utilities
