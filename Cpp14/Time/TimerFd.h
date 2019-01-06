//------------------------------------------------------------------------------
/// \file TimerFd.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  A POSIX timer that delivers expirations by file descriptor (fd).
/// \ref http://man7.org/linux/man-pages/man2/timerfd_create.2.html
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
///  g++ -std=c++17 -I ../ ../Utilities/Errno.cpp \
///   ../Utilities/ErrorHandling.cpp Specifications.cpp Clocks.cpp \
///     TimerFd_main.cpp -o TimerFd_main
//------------------------------------------------------------------------------
#ifndef _TIME_TIMER_FD_H_
#define _TIME_TIMER_FD_H_

#include "Clocks.h" // ClockIDs, check_result
#include "Specifications.h" // IntervalTimerSpecification
#include "Utilities/ErrorHandling.h"
#include "Utilities/casts.h" // get_underlying_value

#include <sys/timerfd.h>
#include <type_traits>
#include <unistd.h>

namespace Time
{

//------------------------------------------------------------------------------
/// \brief enum class for all clock flags, that maybe bitwise ORed in to change
/// behavior of timerfd_create().
//------------------------------------------------------------------------------
enum class ClockFlags : int
{
  default_value = 0,
  non_blocking = TFD_NONBLOCK,
  close_on_execute = TFD_CLOEXEC
};

template <
  ClockIds ClockId = ClockIds::monotonic,
  int clock_flags =
    static_cast<std::underlying_type_t<ClockFlags>>(
      ClockFlags::default_value)
  >
class TimerFd
{
  public:

    TimerFd():
      fd_{::timerfd_create(
        static_cast<std::underlying_type_t<ClockIds>>(ClockId),
        clock_flags)}
    {
      Utilities::ErrorHandling::HandleReturnValue{errno}(
        fd_,
        "create file descriptor (::timerfd_create)");
    }

    ~TimerFd()
    {
      Utilities::ErrorHandling::HandleClose{}(::close(fd_));
    }

  private:

    int fd_;
};

} // namespace Time

#endif // _TIME_TIMER_FD_H_
