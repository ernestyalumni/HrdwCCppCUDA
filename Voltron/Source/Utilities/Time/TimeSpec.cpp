#include "TimeSpec.h"

#include <ostream>

using std::ostream;

namespace Utilities
{
namespace Time
{

TimeSpec::TimeSpec():
  ::timespec{0, 0}
{}

TimeSpec::TimeSpec(
  const long time_value_sec,
  const long time_value_nsec
  ):
  ::timespec{time_value_sec, time_value_nsec}
{}

ostream& operator<<(
  ostream& os,
  const TimeSpec& time_specification)
{
  os << time_specification.tv_sec << ' ' << time_specification.tv_nsec << '\n';
  return os;
}

TimeSpec operator-(const TimeSpec& t1, const TimeSpec& t2)
{
  long seconds {t1.tv_sec - t2.tv_sec};
  long nanoseconds {t1.tv_nsec - t2.tv_nsec};

  if (nanoseconds < 0)
  {
    seconds -= 1;
    nanoseconds += 1'000'000'000;
  }

  return TimeSpec{seconds, nanoseconds};
}

} // namespace Time
} // namespace Utilities
