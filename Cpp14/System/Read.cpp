//------------------------------------------------------------------------------
/// \file Read.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief read() as a C++ functor with CRTP pattern.
/// \ref http://man7.org/linux/man-pages/man2/read.2.html
/// \details Read from a fd.
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
///  g++ -std=c++17 -I ../ Close.cpp ../Utilities/Errno.cpp
///   ../Utilities/ErrorHandling.cpp Close_main.cpp -o Close_main
//------------------------------------------------------------------------------
#include "Read.h"

namespace System
{

Read::Read() = default;

ssize_t Read::operator()(const int fd, void* buf, const size_t count)
{
  const ssize_t result {::read(fd, buf, count)};

  return result;
}

Read::HandleRead::HandleRead() = default;

void Read::HandleRead::operator()(const ssize_t result)
{
  HandleReturnValue::operator()(result, "Read (::read), with error.");
}

} // namespace System