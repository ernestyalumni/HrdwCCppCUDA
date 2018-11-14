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
///  g++ -std=c++14 Clocks_main.cpp -o Clocks_main
//------------------------------------------------------------------------------
#ifndef _UTILITIES_CLOCK_H_
#define _UTILITIES_CLOCK_H_

#include "Chrono.h"
#include "CheckReturn.h" // CheckReturn
#include "casts.h" // get_underlying_value

#include <ctime>
#include <iostream>
//#include <time.h> // ::clock_gettime, ::timespec

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

  const ::timespec* to_timespec() const
  {
    reinterpret_cast<const ::timespec*>(this);
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

template <ClockIDs ClockId = ClockIDs::monotonic>
void get_clock_resolution(TimeSpecification& time_specification)
{
  CheckReturn()(
    ::clock_getres(
    get_underlying_value<ClockIDs>(ClockId),
    time_specification.to_timespec()),
    "Retrieve resolution from clock failed (::clock_getres");
}

template <ClockIDs ClockId = ClockIDs::monotonic>
void get_clock_time(TimeSpecification& time_specification)
{
  CheckReturn()(
    ::clock_gettime(
    get_underlying_value<ClockIDs>(ClockId),
    time_specification.to_timespec()),
    "Retrieve time from clock failed (::clock_gettime");
}

template <ClockIDs ClockId = ClockIDs::monotonic>
void set_clock_time(TimeSpecification& time_specification)
{
  CheckReturn()(
    ::clock_settime(
    get_underlying_value<ClockIDs>(ClockId),
    time_specification.to_timespec()),
    "set time from clock failed (::clock_settime");
}

//------------------------------------------------------------------------------
/// \brief Thin wrapper class for ::clock, to determine processor time.
/// \details clock() function returns approximation of processor time used by
/// program.
/// \ref http://man7.org/linux/man-pages/man3/clock.3.html
//------------------------------------------------------------------------------
class ProcessorTimeClock
{
  public:

    ProcessorTimeClock():
      last_processor_time_obtained_{get_processor_time()}
    {}

    auto clock_ticks_per_second() const
    {
      return clock_ticks_per_second_;
    }

    clock_t operator()()
    {
      last_processor_time_obtained_ = get_processor_time();
      return last_processor_time_obtained_;
    }

    clock_t last_processor_time_obtained() const
    {
      return last_processor_time_obtained_;
    }

    //--------------------------------------------------------------------------
    /// \ref https://en.cppreference.com/w/c/chrono/clock_t
    //--------------------------------------------------------------------------
    clock_t elapsed_time()
    {
      const clock_t present_time {get_processor_time()};
      const clock_t cpu_time_used {present_time - last_processor_time_obtained()};

      this->operator()();

      return cpu_time_used;
    }

  protected:

    void set_last_processor_time_obtained(const clock_t& t)
    {
      last_processor_time_obtained_ = t;
    }

  private:

    clock_t get_processor_time()
    {
      return ::clock();
    }

    static constexpr auto clock_ticks_per_second_ = CLOCKS_PER_SEC;

    clock_t last_processor_time_obtained_;
};

} // namespace Utilities

#endif // _UTILITIES_CLOCK_H_
