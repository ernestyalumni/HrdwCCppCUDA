//------------------------------------------------------------------------------
/// \file ErrorHandling.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief Error handling C++ functors to check POSIX Linux system call
/// results.
//------------------------------------------------------------------------------
#ifndef UTILITIES_ERROR_HANDLING_HANDLE_RETURN_VALUE_H
#define UTILITIES_ERROR_HANDLING_HANDLE_RETURN_VALUE_H

#include "GetErrorNumber.h"

#include <optional>
#include <string>

namespace Utilities
{
namespace ErrorHandling
{

class HandleReturnValue
{
  public:

    using OptionalErrorNumber = std::optional<int>;

    HandleReturnValue():
      get_error_number_{},
      return_value_{}
    {}

    //--------------------------------------------------------------------------
    /// \details If the return_value is less than 0, the error number is
    /// obtained (since the errno would have been set), and a non-empty optional
    /// is returned. Otherwise, an empty optional is returned.
    //--------------------------------------------------------------------------
    virtual OptionalErrorNumber operator()(const int return_value);

    inline int error_number() const
    {
      return get_error_number_.error_number();
    }

    inline int return_value() const
    {
      return return_value_;
    }


    inline std::string as_string() const
    {
      return get_error_number_.as_string();
    }

  private:

    GetErrorNumber get_error_number_;
    int return_value_;
};

} // namespace ErrorHandling
} // namespace Utilities

#endif // UTILITIES_ERROR_HANDLING_HANDLE_RETURN_VALUE_H
