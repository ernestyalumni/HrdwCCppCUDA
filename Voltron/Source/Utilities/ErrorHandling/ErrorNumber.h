//------------------------------------------------------------------------------
/// \file ErrorNumber.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Wrappers and examples classes for errno and error handling.
/// \ref https://en.cppreference.com/w/cpp/error/errc
/// https://en.cppreference.com/w/cpp/error/errno_macros
/// \details Scoped enumeration (enum class) std::errc defines values of
/// portable error conditions corresponding to POSIX error codes.
//------------------------------------------------------------------------------
#ifndef UTILITIES_ERROR_HANDLING_ERROR_NUMBER_H
#define UTILITIES_ERROR_HANDLING_ERROR_NUMBER_H

#include <cerrno> // E2BIG, EACCESS, ...
#include <cstring> // std::strerror
#include <string>
#include <system_error> // std::errc, std::make_error_code, std::error_code

namespace Utilities
{
namespace ErrorHandling
{

//------------------------------------------------------------------------------
/// \class ErrorNumbers
/// \name ErrorNumbers
/// \brief Wraps POSIX error codes macros.
/// \details Each of the macros expands to integer constant expressions of type
/// int, each with positive value, matching most POSIX error codes.
//------------------------------------------------------------------------------
enum class ErrorNumbers;

//------------------------------------------------------------------------------
/// \brief Wrapper class for errno, or error numbers.
//------------------------------------------------------------------------------
class ErrorNumber
{
  public:

    ErrorNumber();

    explicit ErrorNumber(const int error_number);

    explicit ErrorNumber(const std::error_code& error_code);

    /// \ref https://en.cppreference.com/w/cpp/error/error_code/error_code
    ErrorNumber(const int error_number, const std::error_category& ecat);

    // Accessors
    /// \ref https://stackoverflow.com/questions/44935159/why-must-accessor-functions-be-const-where-is-the-vulnerability

    /// \ref https://en.cppreference.com/w/cpp/string/byte/strerror
    std::string as_string() const
    {
      return std::string{std::strerror(error_number_)};
    }

    int error_number() const
    {
      return error_number_;
    }

    std::error_code error_code() const
    {
      return error_code_;
    }

    std::error_condition error_condition() const
    {
      return error_condition_;
    }

  private:

    int error_number_;

    std::error_code error_code_;

    /// \ref https://en.cppreference.com/w/cpp/error/error_condition/error_condition
    std::error_condition error_condition_;
};

} // namespace ErrorHandling
} // namespace Utilities

#endif // UTILITIES_ERROR_HANDLING_ERROR_NUMBER_H
