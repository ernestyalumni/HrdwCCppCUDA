//------------------------------------------------------------------------------
/// \file Close.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief ::close as a C++ functor source file.
/// \ref http://man7.org/linux/man-pages/man2/close.2.html
/// \details Closes a fd.
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
/// g++ -std=c++17 -I ../ Close.cpp ../Utilities/Errno.cpp
///   ../Utilities/ErrorHandling.cpp Close_main.cpp -o Close_main
//------------------------------------------------------------------------------
#include "Close.h"
#include "Utilities/Errno.h"
#include "Utilities/casts.h" // get_underlying_value

#include <iostream>
#include <system_error>
#include <unistd.h> // ::close

using Utilities::ErrorHandling::Details::ErrorNumbers;
using Utilities::get_underlying_value;

namespace System
{

//------------------------------------------------------------------------------
/// \details fd is a nonnegative number. Start with -1.
//------------------------------------------------------------------------------
Close::Close():
  fd_{-1}
{}

void Close::operator()(const int fd)
{
  // Don't try to ::close an fd that was closed before with Close.
  if (fd != fd_)
  {
    HandleClose()(::close(fd));
    fd_ = fd;
  }
  else
  {
    std::cerr << "Attempting to close a previously closed fd.\n";
  }
}

Close::HandleClose::HandleClose() = default;

void Close::HandleClose::operator()(const int result)
{
  if (result < 0)
  {
    get_error_number();

    const int error_number {this->error_number().error_number()};

    if (error_number == get_underlying_value<ErrorNumbers>(ErrorNumbers::ebadf))
    {
      throw std::system_error(
        error_number,
        std::system_category(),
        "Failed to ::close(fd) with errno : " +
          this->error_number().as_string() + " and error number " +
            std::to_string(this->error_number().error_number()) +
              "for EBADF, fd isn't a valid open fd\n");
    }
    else if (error_number ==
      get_underlying_value<ErrorNumbers>(ErrorNumbers::eintr))
    {
      throw std::system_error(
        error_number,
        std::system_category(),
        "Failed to ::close(fd) with errno : " +
          this->error_number().as_string() +
            " and error number " +
              std::to_string(this->error_number().error_number()) +
                "for EINTR, ::close() call interrupted by a signal\n");
    }
    else if (error_number ==
      get_underlying_value<ErrorNumbers>(ErrorNumbers::eio))
    {
      throw std::system_error(
        error_number,
        std::system_category(),
        "Failed to ::close(fd) with errno : " +
          this->error_number().as_string() +
            " and error number " +
              std::to_string(this->error_number().error_number()) +
                "for EIO, I/O error occured\n");
    }
    else if (error_number ==
      get_underlying_value<ErrorNumbers>(ErrorNumbers::enospc))
    {
      throw std::system_error(
        error_number,
        std::system_category(),
        "Failed to ::close(fd) with errno : " +
          this->error_number().as_string() +
            " and error number " +
              std::to_string(this->error_number().error_number()) +
                "for ENOSPC, no space left; on NFS, error reported after first write" +
                  "\n");
    }
    else if (error_number == EDQUOT)
    {
      throw std::system_error(
        error_number,
        std::system_category(),
        "Failed to ::close(fd) with errno : " +
          this->error_number().as_string() +
            " and error number " +
              std::to_string(this->error_number().error_number()) +
                "for EDQUOT, no space left; on NFS, error reported after first write" +
                  "\n");
    }
    else // error unexpected
    {
      throw std::system_error(
        errno,
        std::system_category(),
        "Failed to ::close(fd) with errno : " +
          this->error_number().as_string() +
            " and error number " +
              std::to_string(this->error_number().error_number()) + '\n');
    }
  }
}

} // namespace System  