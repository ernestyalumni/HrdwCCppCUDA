//------------------------------------------------------------------------------
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref
/// \details
//------------------------------------------------------------------------------
#include "HandleReturnValue.h"

#include "ErrorNumber.h" // ErrorNumber

#include <optional>
#include <string>
#include <system_error> // std::system_error, std::system_category

using std::string;

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

ThrowSystemErrorOnNegativeReturnValue::ThrowSystemErrorOnNegativeReturnValue() :
  HandleError{},
  custom_error_message_{}
{}

ThrowSystemErrorOnNegativeReturnValue::ThrowSystemErrorOnNegativeReturnValue(
  const string& custom_error_message
  ):
  HandleError{},
  custom_error_message_{custom_error_message}
{}

void ThrowSystemErrorOnNegativeReturnValue::operator()(const int result)
{
  handle_negative_one_result(result);
}

void ThrowSystemErrorOnNegativeReturnValue::handle_negative_one_result(
  const int result)
{
  if (result < 0)
  {
    get_error_number();

    string error_message_to_send {
      std::to_string(error_number().error_number()) + '\n'};

    if (!custom_error_message_.empty())
    {
      error_message_to_send =
        custom_error_message_ +
        "; Error Message: " +
        error_message_to_send;
    }

    throw std::system_error(
      errno,
      std::system_category(),
      error_message_to_send);
  }
}


} // namespace ErrorHandling
} // namespace Utilities