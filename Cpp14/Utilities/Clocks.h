//------------------------------------------------------------------------------
/// \file Clocks.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  A POSIX clock(s).
/// \ref https://linux.die.net/man/3/clock_gettime     
/// \details 
/// \copyright If you find this code useful, feel free to donate directly and
/// easily at this direct PayPal link: 
///
/// https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
/// 
/// which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
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
///  g++ -std=c++14 -lrt ../MessageQueue_main.cpp -o ../MessageQueue_main
//------------------------------------------------------------------------------
#ifndef _UTILITIES_CLOCK_H_
#define _UTILITIES_CLOCK_H_

#include "Chrono.h"

#include <cstring> // strerror
#include <iostream>
#include <system_error> 
#include <time.h> // ::clock_gettime, ::timespec

namespace Utilities
{

//------------------------------------------------------------------------------
/// \brief enum class for all clock id's, type of clock.
/// \details clockid argument specifies clock that's used to mark the progress
/// of the timer
/// \ref http://man7.org/linux/man-pages/man2/timerfd_create.2.html
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

  // Used by ::timerfd_create exclusively.
  boot_time = CLOCK_BOOTTIME,
  real_time_alarm = CLOCK_REALTIME_ALARM,
  boot_time_alarm = CLOCK_BOOTTIME_ALARM
};

//------------------------------------------------------------------------------
/// struct timespec
/// {
///   time_t tv_sec; // Seconds
///   long tv_nsec; // Nanoseconds
/// };
//------------------------------------------------------------------------------
struct TimeSpecification : public ::timespec 
{
  using Nanoseconds = Utilities::Nanoseconds;
  using Seconds = Utilities::Nanoseconds;

  explicit TimeSpecification(
    const long time_value_sec,
    const long time_value_nsec = 0):
    ::timespec{time_value_sec, time_value_nsec}  
  {}

  TimeSpecification():
    TimeSpecification{0, 0}
  {}

  template <class Duration>
  explicit TimeSpecification(const Duration& duration)
  {
    const Seconds duration_secs {Utilities::duration_cast<Seconds>(duration)};
    const Nanoseconds duration_nanosecs {
      Utilities::duration_cast<Nanoseconds>(duration - duration_secs)};

    tv_sec = duration_secs.count();
    tv_nsec = duration_nanosecs.count();
  }

  ::timespec* to_timespec()
  {
    reinterpret_cast<::timespec*>(this);
  }

  friend std::ostream& operator<<(std::ostream& os,
    const TimeSpecification& time_specification);
};

std::ostream& operator<<(std::ostream& os,
  const TimeSpecification& time_specification)
{
  os << time_specification.tv_sec << ' ' << time_specification.tv_nsec << '\n';
}

//------------------------------------------------------------------------------
// \details Specified by Linux manual page, in that system calls that include
// - ::clock_gettime
// - ::clock_settime
// - ::clock_getres
// are documented to return 0 for success, or -1 for failure (in which case
// errno is set appropriately), and that errno will be returned by this
// function.
//------------------------------------------------------------------------------
int check_result(int e, const std::string& custom_error_string)
{
  if (e < 0)
  {
    std::cout << " errno : " << std::strerror(errno) << '\n';
    throw std::system_error(
      errno,
      std::generic_category(),
      "Failed to " + custom_error_string + "\n");
  }
  return errno;
}

template <ClockIDs clock_id = ClockIDs::monotonic>
void get_clock_resolution(TimeSpecification& time_specification)
{
  check_result(
    ::clock_getres(
    static_cast<int>(clock_id),
    time_specification.to_timespec()),
    "Retrieve resolution from clock failed (::clock_getres");
}

template <ClockIDs clock_id = ClockIDs::monotonic>
void get_clock_time(TimeSpecification& time_specification)
{
  check_result(
    ::clock_gettime(
    static_cast<int>(clock_id),
    time_specification.to_timespec()),
    "Retrieve time from clock failed (::clock_gettime");
}

template <ClockIDs clock_id = ClockIDs::monotonic>
void set_clock_time(TimeSpecification& time_specification)
{
  check_result(
    ::clock_settime(
    static_cast<int>(clock_id),
    time_specification.to_timespec()),
    "set time from clock failed (::clock_settime");
}

} // namespace Utilities

#endif // _UTILITIES_CLOCK_H_
