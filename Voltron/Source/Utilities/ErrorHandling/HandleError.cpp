//------------------------------------------------------------------------------
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref
/// \details
//------------------------------------------------------------------------------
#include "HandleError.h"

#include "ErrorNumber.h" // ErrorNumber

#include <system_error> // std::errc, std::make_error_code, std::error_code

namespace Utilities
{
namespace ErrorHandling
{

HandleError::HandleError() :
  error_number_{}
{}

HandleError::HandleError(const int error_number) :
  error_number_{error_number}
{}

HandleError::HandleError(const std::error_code& error_code) :
  error_number_{error_code}
{}

HandleError::HandleError(const std::system_error& err) :
  error_number_{err}
{}


void HandleError::get_error_number()
{  
  error_number_ = ErrorNumber{};
}

void HandleError::operator()()
{
  get_error_number();
}

} // namespace ErrorHandling
} // namespace Utilities
