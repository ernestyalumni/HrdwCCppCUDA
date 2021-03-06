//------------------------------------------------------------------------------
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref
/// \details
//------------------------------------------------------------------------------
#include "HandleReturnValue.h"

#include "ErrorNumber.h" // ErrorNumber

#include <optional>

namespace Utilities
{
namespace ErrorHandling
{

HandleReturnValue::HandleReturnValue() :
  HandleError{}
{}

void HandleReturnValue::operator()(const int result)
{
  handle_negative_one_result(result);
}

void HandleReturnValue::handle_negative_one_result(const int result)
{
  if (result < 0)
  {
    get_error_number();
  }
}

HandleReturnValueWithOptional::HandleReturnValueWithOptional() :
  HandleError{}
{}

HandleReturnValueWithOptional::OptionalErrorNumber
  HandleReturnValueWithOptional::operator()(const int result)
{
  return handle_result(result);
}

HandleReturnValueWithOptional::OptionalErrorNumber
  HandleReturnValueWithOptional::handle_result(const int result)
{
  if (result < 0)
  {
    get_error_number();

    return std::make_optional<ErrorNumber>(error_number());
  }
  else
  {
    return std::nullopt;
  }
}


} // namespace ErrorHandling
} // namespace Utilities