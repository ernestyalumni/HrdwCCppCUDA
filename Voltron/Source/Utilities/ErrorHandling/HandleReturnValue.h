//------------------------------------------------------------------------------
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
//------------------------------------------------------------------------------
#ifndef UTILITIES_ERROR_HANDLING_HANDLE_RETURN_VALUE_H
#define UTILITIES_ERROR_HANDLING_HANDLE_RETURN_VALUE_H

#include "HandleError.h"

#include <optional>
#include <string>

namespace Utilities
{
namespace ErrorHandling
{

//------------------------------------------------------------------------------
/// \brief Virtual base class for C++ functor for checking the result of a
/// POSIX Linux system calls, usually when creating a new file descriptor.
//------------------------------------------------------------------------------
class HandleReturnValue : public HandleError
{
  public:

    HandleReturnValue();

    // TODO: Copy this over to the derived classes.
    //explicit HandleReturnValue(const std::string& custom_error_string);

    //--------------------------------------------------------------------------
    /// \brief Effectively run errno macro and save results in an ErrorNumber.
    //--------------------------------------------------------------------------   
    virtual void operator()(const int result);

  protected:

    using HandleError::get_error_number;

    virtual void handle_negative_one_result(const int result);
};    

class HandleReturnValueWithOptional : public HandleError
{
  public:

    using OptionalErrorNumber = std::optional<ErrorNumber>;

    HandleReturnValueWithOptional();

    virtual OptionalErrorNumber operator()(const int result);

  protected:

    virtual OptionalErrorNumber handle_result(const int result);
};

} // namespace ErrorHandling
} // namespace Utilities

#endif // UTILITIES_ERROR_HANDLING_HANDLE_ERROR_H
