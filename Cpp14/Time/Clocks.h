//------------------------------------------------------------------------------
/// \file Clocks.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  A POSIX clock(s).
/// \ref https://linux.die.net/man/3/clock_gettime
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
///     Clocks_main.cpp-o Clocks_main
//------------------------------------------------------------------------------
#ifndef _TIME_CLOCKS_H_
#define _TIME_CLOCKS_H_

#include "Specifications.h"
#include "Utilities/ErrorHandling.h"
#include "Utilities/casts.h" // get_underlying_value

#include <ctime> // CLOCK_REALTIME, CLOCK_MONOTONIC, ..., ::timespec

namespace Time
{

//------------------------------------------------------------------------------
/// \brief enum class for all clock id's, type of clock.
/// \details clockid argument specifies clock that's used to mark the progress
/// of the timer
/// \ref https://www.systutorials.com/docs/linux/man/2-clock_gettime/
/// http://man7.org/linux/man-pages/man2/timerfd_create.2.html
/// https://linux.die.net/man/3/clock_gettime
//------------------------------------------------------------------------------
enum class ClockIds : int
{
  // Used by both ::clock_gettime and ::timerfd_create
  real_time = CLOCK_REALTIME,
  monotonic = CLOCK_MONOTONIC,

  // Used by ::clock_gettime exclusively.
  process_cpu_time = CLOCK_PROCESS_CPUTIME_ID,
  thread_cpu_time = CLOCK_THREAD_CPUTIME_ID,

  monotonic_raw = CLOCK_MONOTONIC_RAW,

  real_time_coarse = CLOCK_REALTIME_COARSE,
  monotonic_coarse = CLOCK_MONOTONIC_COARSE,

  // Used by ::timerfd_create exclusively.
  boot_time = CLOCK_BOOTTIME,
  real_time_alarm = CLOCK_REALTIME_ALARM,
  boot_time_alarm = CLOCK_BOOTTIME_ALARM
};

//------------------------------------------------------------------------------
/// \brief ::clock_gettime wrapper as a C++ functor.
//------------------------------------------------------------------------------
template <ClockIds ClockId = ClockIds::monotonic>
class GetClockTime
{
  public:

    GetClockTime():
      time_specification_{}
    {}

    TimeSpecification operator()()
    {
      Utilities::ErrorHandling::HandleReturnValue()(
        ::clock_gettime(
          Utilities::get_underlying_value<ClockIds>(ClockId),
          time_specification_.to_timespec_pointer()),
          "Retrieve time from clock failed (::clock_gettime");

      return time_specification();
    }

    TimeSpecification time_specification() const
    {
      return time_specification_;
    }

  private:

    TimeSpecification time_specification_;
};

} // namespace Time

#endif // _TIME_CLOCK_H_
