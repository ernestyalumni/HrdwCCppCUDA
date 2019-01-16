//------------------------------------------------------------------------------
/// \file Specifications.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  A POSIX time specifications source file.
/// \ref https://linux.die.net/man/3/clock_gettime
/// http://pubs.opengroup.org/onlinepubs/7908799/xsh/time.h.html
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
///  g++ -std=c++17 -I ../ Specifications_main.cpp Specifications.cpp -o Specifications_main
//------------------------------------------------------------------------------
#include "Specifications.h"

#include "Utilities/Chrono.h" // Seconds, Nanoseconds, duration_cast

#include <ctime> // CLOCK_REALTIME, CLOCK_MONOTONIC, ..., ::timespec
#include <ostream>
#include <stdexcept> // std::invalid_argument

using Utilities::Nanoseconds;
using Utilities::Seconds;
using Utilities::duration_cast;

namespace Time
{

::timespec carry_nanoseconds_to_seconds(const ::timespec& time_spec)
{
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

TimeSpecification::TimeSpecification(const ::timespec& timespec):
  timespec_{timespec}
{}

bool TimeSpecification::operator>=(const TimeSpecification& rhs) const
{
  if (timespec_.tv_sec > rhs.timespec_.tv_sec)
  {
    return true;
  }
  else if (timespec_.tv_sec == rhs.timespec_.tv_sec)
  {
    return (timespec_.tv_nsec >= rhs.timespec_.tv_nsec);
  }
  return false;
}

bool TimeSpecification::operator==(const TimeSpecification& rhs) const
{
  return ((timespec_.tv_sec == rhs.timespec_.tv_sec) &&
    (timespec_.tv_nsec == rhs.timespec_.tv_nsec));
}

TimeSpecification TimeSpecification::operator-(
  const TimeSpecification& rhs) const
{
  if (!(*this >= rhs))
  {
    throw std::invalid_argument("Cannot subtract by a later time.");
  }

  const time_t sec {timespec_.tv_sec - rhs.timespec_.tv_sec};
  const long nsec {timespec_.tv_nsec - rhs.timespec_.tv_nsec};

  return TimeSpecification{
    carry_nanoseconds_to_seconds(::timespec{sec, nsec})};
}

std::ostream& operator<<(std::ostream& os,
  const TimeSpecification& time_specification)
{
  os << time_specification.timespec_.tv_sec << ' ' <<
    time_specification.timespec_.tv_nsec << '\n';
  return os;
}

IntervalTimerSpecification::IntervalTimerSpecification():
  itimerspec_{0, 0, 0, 0}
{}

IntervalTimerSpecification::IntervalTimerSpecification(
  const ::itimerspec i_timer_spec
  ):
  itimerspec_{i_timer_spec}
{}


IntervalTimerSpecification::IntervalTimerSpecification(
  const TimeSpecification& interval_time_specification,
  const TimeSpecification& initial_expiration_time_specification
  ):
  itimerspec_{
    ::timespec(interval_time_specification),
    ::timespec(initial_expiration_time_specification)}
{}

IntervalTimerSpecification::IntervalTimerSpecification(
  const TimeSpecification& time_specification):
  IntervalTimerSpecification{time_specification, time_specification}
{}

std::ostream& operator<<(std::ostream& os,
  const IntervalTimerSpecification& interval_timer_specification)
{
  os <<
    TimeSpecification{interval_timer_specification.itimerspec_.it_interval} <<
    ' ' <<
    TimeSpecification{interval_timer_specification.itimerspec_.it_value} <<
    '\n';

  return os;
}

} // namespace Time
