//------------------------------------------------------------------------------
/// \file Fork.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief ::fork as a C++ functor source file.
/// \ref http://man7.org/linux/man-pages/man2/fork.2.html
/// \details Create a child process.
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
/// g++ -std=c++17 -I ../ Fork.cpp ../Utilities/Errno.cpp
///   ../Utilities/ErrorHandling.cpp Fork_main.cpp -o Fork_main
//------------------------------------------------------------------------------
#include "Fork.h"

#include <unistd.h> // ::fork

namespace System
{

Fork::Fork() = default;

pid_t Fork::operator()()
{
  process_id_ = ::fork();

  HandleFork()(process_id_);

  return process_id_;
}

Fork::HandleFork::HandleFork() = default;

void Fork::HandleFork::operator()(const pid_t result)
{
  this->operator()(result, "Fork process (::fork)");
}

} // namespace System  