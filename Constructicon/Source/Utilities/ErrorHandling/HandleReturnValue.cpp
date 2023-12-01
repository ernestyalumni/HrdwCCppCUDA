//------------------------------------------------------------------------------
/// \file HandleReturnValue.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief Source file for error handling C++ functors to check POSIX Linux
/// system call results.
//------------------------------------------------------------------------------
#include "HandleReturnValue.h"

#include <optional>

namespace Utilities
{
namespace ErrorHandling
{

HandleReturnValue::OptionalErrorNumber HandleReturnValue::operator()(
  const int return_value)
{
  return_value_ = return_value;

  if (return_value < 0)
  {
    get_error_number_();

    return std::make_optional<int>(get_error_number_.error_number());
  }
  else
  {
    return std::nullopt;
  }
}

} // namespace ErrorHandling
} // namespace Utilities
