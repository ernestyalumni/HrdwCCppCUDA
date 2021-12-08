#ifndef UTILITIES_TIME_CLOCK_ID_H
#define UTILITIES_TIME_CLOCK_ID_H

#include <ctime> // CLOCK_REALTIME, CLOCK_MONOTONIC, ...

namespace Utilities
{
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
enum class ClockId : int
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

} // namespace Time
} // namespace Utilities

#endif // UTILITIES_TIME_CLOCK_ID_H
