#ifndef UTILITIES_ERROR_HANDLING_GET_ERROR_NUMBER_H
#define UTILITIES_ERROR_HANDLING_GET_ERROR_NUMBER_H

#include <cerrno>
#include <cstring> // std::strerror

namespace Utilities
{
namespace ErrorHandling
{


//------------------------------------------------------------------------------
/// \brief Get the error number from the errno variable.
/// \details Wrapper class for errno, or error numbers.
//------------------------------------------------------------------------------
class GetErrorNumber
{
  public:

    //--------------------------------------------------------------------------
    /// \brief Default constructor.
    /// \details Provides important feature of using the preprocessor macro
    /// errno used for error indication. Library functions and implementation-
    /// dependent functions are allowed to write positive integers to errno
    /// whether or not an error occurred.
    //--------------------------------------------------------------------------
    GetErrorNumber():
      error_number_{errno}
    {}

    void get_error_number()
    {
      error_number_ = errno;
    }

    //--------------------------------------------------------------------------
    /// \details std::strerror returns pointer to textual description of system
    /// error code errnum, identical to description that would be printed by
    /// std::perror().
    /// errnum usually acquired from errno variable.
    /// \ref https://en.cppreference.com/w/cpp/string/byte/strerror
    //--------------------------------------------------------------------------
    std::string as_string() const
    {
      return std::string{std::strerror(error_number_)};
    }


  private:

    int error_number_;  
};

} // namespace ErrorHandling
} // namespace Utilities

#endif // UTILITIES_ERROR_HANDLING_ERROR_NUMBER_H
