//------------------------------------------------------------------------------
/// \file ErrorHandling.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief Source file for error handling C++ functors to check POSIX Linux
/// system call results.
/// \ref
/// \details
//------------------------------------------------------------------------------
#include "ErrorHandling.h"

#include <iostream>
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

void HandleReturnValue::get_error_number()
{  
  error_number_ = ErrorNumber{};
}

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

void HandleClose::operator()(const int result)
{
  this->operator()(result, "Failed to close fd (::close())");
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