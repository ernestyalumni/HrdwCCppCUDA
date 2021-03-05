//------------------------------------------------------------------------------
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref
/// \details
//------------------------------------------------------------------------------
#include "HandleReturnValue.h"

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
  if (result < -1)
  {
    get_error_number();
  }
}

} // namespace ErrorHandling
} // namespace Utilities