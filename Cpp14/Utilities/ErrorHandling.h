//------------------------------------------------------------------------------
/// \file ErrorHandling.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief Error handling C++ functors to check POSIX Linux system call
/// results.
/// \ref
/// \details
/// \copyright If you find this code useful, feel free to donate directly
/// (username ernestyalumni or email address above), going directly to:
///
/// paypal.me/ernestyalumni
///
/// which won't go through a 3rd. party like indiegogo, kickstarter, patreon.
/// Otherwise, I receive emails and messages on how all my (free) material on
/// physics, math, and engineering have helped students with their studies, and
/// I know what it's like to not have money as a student, but love physics (or
/// math, sciences, etc.), so I am committed to keeping all my material
/// open-source and free, whether or not sufficiently crowdfunded, under the
/// open-source MIT license: feel free to copy, edit, paste, make your own
/// versions, share, use as you wish.
/// Peace out, never give up! -EY
//------------------------------------------------------------------------------
/// COMPILATION TIPS:
///  g++ -std=c++17 Errno.cpp ErrorHandling.cpp ErrorHandling_main.cpp -o \
///   ErrorHandling_main
//------------------------------------------------------------------------------
#ifndef _UTILITIES_ERROR_HANDLING_ERROR_HANDLING_H_
#define _UTILITIES_ERROR_HANDLING_ERROR_HANDLING_H_

#include "Errno.h" // ErrorNumber

#include <string> // std::string

namespace Utilities
{
namespace ErrorHandling
{

//------------------------------------------------------------------------------
/// \brief Virtual base class for C++ functor for checking the result of a
/// POSIX Linux system calls, usually when creating a new file descriptor.
//------------------------------------------------------------------------------
class HandleReturnValue
{
  public:

    HandleReturnValue();

    HandleReturnValue(const int error_number);

    virtual void operator()(
      const int result,
      const std::string& custom_error_string);

    virtual void operator()(const int result);

    void get_error_number()
    {
      error_number_ = ErrorNumber{};
    }

    ErrorNumber error_number() const
    {
      return error_number_;
    }

  private:

    ErrorNumber error_number_;
};

//------------------------------------------------------------------------------
// \ref http://man7.org/linux/man-pages/man2/close.2.html
// \details On error, -1 is returned, and errno is set appropriately.
// 0 indicates success.
//------------------------------------------------------------------------------
class HandleClose : public HandleReturnValue
{
  public:

    HandleClose() = default;

    void operator()(const int result);

  private:

    using HandleReturnValue::operator();
};

} // namespace ErrorHandling
} // namespace Utilities

#endif // _UTILITIES_ERROR_HANDLING_ERROR_HANDLING_H_
