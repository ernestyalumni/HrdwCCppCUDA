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
///  g++ -std=c++17 -I ../ Clocks_main.cpp -o Clocks_main
//------------------------------------------------------------------------------
#ifndef _TIME_CLOCKS_H_
#define _TIME_CLOCKS_H_

#if 0
#include "CheckReturn.h" // CheckReturn
#include "casts.h" // get_underlying_value

#include "Specifications.h"

#include <iostream>
#include <stdexcept> // std::runtime_error
#include <type_traits>
//#include <time.h> // ::clock_gettime, ::timespec
#endif

#include "Utilities/Chrono.h" // Seconds, Nanoseconds, duration_cast

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
enum class ClockIDs : int
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
/// \name carry_nanoseconds_to_seconds
/// \brief Carry seconds from tv_nsec field in ::timespec struct into tv_sec
/// field.
/// \details Works for even negative nanoseconds.
//------------------------------------------------------------------------------
::timespec carry_nanoseconds_to_seconds(const ::timespec& time_spec)
{
  using Utilities::Nanoseconds;
  using Utilities::Seconds;
  using Utilities::duration_cast;

  Seconds seconds {time_spec.tv_sec};
  Nanoseconds nanoseconds {time_spec.tv_nsec};

  const Seconds carry_from_nanoseconds {duration_cast<Seconds>(nanoseconds)};

  seconds += carry_from_nanoseconds;
  nanoseconds -= duration_cast<Nanoseconds>(carry_from_nanoseconds);

  if (nanoseconds < Nanoseconds{0})
  {
    // "borrow" or subtract 1 second from seconds.
    seconds -= Seconds{1};
    nanoseconds += duration_cast<Nanoseconds>(Seconds{1});
  }

  return ::timespec {seconds.count(), nanoseconds.count()};
}


} // namespace Time

#endif // _TIME_CLOCK_H_
