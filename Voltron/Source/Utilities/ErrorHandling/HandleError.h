//------------------------------------------------------------------------------
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
//------------------------------------------------------------------------------
#ifndef UTILITIES_ERROR_HANDLING_HANDLE_ERROR_H
#define UTILITIES_ERROR_HANDLING_HANDLE_ERROR_H

#include "ErrorNumber.h" // ErrorNumber

#include <system_error> // std::errc, std::make_error_code, std::error_code

namespace Utilities
{
namespace ErrorHandling
{

class HandleError
{
  public:

    HandleError();

    explicit HandleError(const int error_number);

    explicit HandleError(const std::error_code& error_code);

    explicit HandleError(const std::system_error& err);

    //--------------------------------------------------------------------------
    /// \brief Effectively run errno macro and save results in an ErrorNumber.
    //--------------------------------------------------------------------------   
    virtual void operator()();

    ErrorNumber error_number() const
    {
      return error_number_;
    }

  protected:

    //--------------------------------------------------------------------------
    /// \brief Effectively run errno macro and save results in an ErrorNumber.
    //--------------------------------------------------------------------------
    void get_error_number();

  private:

    ErrorNumber error_number_;
};    


} // namespace ErrorHandling
} // namespace Utilities

#endif // UTILITIES_ERROR_HANDLING_HANDLE_ERROR_H
