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

#include "ErrorNumber.h" // ErrorNumber

#include <iostream>
#include <optional>
#include <string>
#include <system_error> // std::system_error, std::system_category

namespace Utilities
{
namespace ErrorHandling
{

HandleReturnValuePassively::HandleReturnValuePassively() = default;

HandleReturnValuePassively::OptionalErrorNumber
  HandleReturnValuePassively::operator()(const int return_value)
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

/*
std::optional<ErrorNumber> HandleClose::operator()(const int return_value)
{
  if (return_value < 0)
  {
    get_error_number();
    
    std::cerr << "Failed to close fd (::close()) with errno: " <<
      error_number().as_string() << " and error number " <<
      std::to_string(error_number().error_number()) << "\n";

    return std::make_optional<ErrorNumber>(error_number());
  }

  return std::nullopt;
}
*/

/*
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
*/

} // namespace ErrorHandling
} // namespace Utilities
