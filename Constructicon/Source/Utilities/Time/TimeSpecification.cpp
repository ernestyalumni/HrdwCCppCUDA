#include "TimeSpecification.h"
#include "Utilities/Time/Chrono.h"

#include <cassert>
#include <chrono>
#include <ctime> // ::timespec
#include <ostream>

using Utilities::Time::Nanoseconds;
using Utilities::Time::Seconds;
using std::ostream;

namespace Utilities
{

namespace Time
{

namespace Details
{

constexpr auto nanoseconds_in_second = 1'000'000'000;

::timespec carry_or_borrow_nanoseconds(const ::timespec& time_specification)
{
  ::timespec result {time_specification};
  const Nanoseconds nanoseconds {time_specification.tv_nsec};

  // Since duration_cast preserves the sign of what it's casting, then we can
  // combine both cases arithmetically.

  // Carry to seconds or "borrow" from seconds, respectively.
  if (nanoseconds >= Nanoseconds{nanoseconds_in_second} ||
    nanoseconds < Nanoseconds{0})
  {
    const Seconds carry_or_borrow_seconds {
      std::chrono::floor<Seconds>(nanoseconds)};

    result.tv_sec += carry_or_borrow_seconds.count();
    result.tv_nsec -= (std::chrono::duration_cast<Nanoseconds>(
      carry_or_borrow_seconds)).count();
  }

  return result;
}

TimeSpecification carry_or_borrow_nanoseconds(
  const TimeSpecification& time_specification)
{
  return TimeSpecification{
    carry_or_borrow_nanoseconds(time_specification.get_timespec())};
}

} // namespace Details

TimeSpecification::TimeSpecification():
  time_specification_{0, 0}
{}

TimeSpecification::TimeSpecification(
  const long time_value_sec,
  const long time_value_nsec):
  time_specification_{time_value_sec, time_value_nsec}
{}

TimeSpecification::TimeSpecification(const ::timespec& timespec):
  time_specification_{timespec}
{}

bool TimeSpecification::operator>=(const TimeSpecification& rhs) const
{
  const ::timespec lhs {
    Details::carry_or_borrow_nanoseconds(this->time_specification_)};

  const ::timespec rhs_ts {
    Details::carry_or_borrow_nanoseconds(rhs.time_specification_)};

  if (lhs.tv_sec == rhs_ts.tv_sec)
  {
    return (lhs.tv_nsec >= rhs_ts.tv_nsec);
  }

  return (lhs.tv_sec > rhs_ts.tv_sec);
}

TimeSpecification TimeSpecification::operator-(const TimeSpecification& rhs)
  const
{
  assert(rhs.time_specification_.tv_nsec >= 0);
  assert(this->time_specification_.tv_nsec >= 0);
  assert(rhs.time_specification_.tv_nsec < Details::nanoseconds_in_second);
  assert(this->time_specification_.tv_nsec < Details::nanoseconds_in_second);

  ::timespec difference {
    time_specification_.tv_sec - rhs.time_specification_.tv_sec,
    time_specification_.tv_nsec - rhs.time_specification_.tv_nsec};

  if (difference.tv_nsec < 0)
  {
    difference.tv_sec -= 1;
    difference.tv_nsec += Details::nanoseconds_in_second;
  }

  return TimeSpecification{difference};
}

ostream& operator<<(
  ostream& os,
  const TimeSpecification& time_specification)
{
  os << time_specification.time_specification_.tv_sec <<
    ' ' <<
    time_specification.time_specification_.tv_nsec <<
    "\n";

  return os;
}

} // namespace Time
} // namespace Utilities