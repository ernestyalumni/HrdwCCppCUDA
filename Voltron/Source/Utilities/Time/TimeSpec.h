#ifndef UTILITIES_TIME_TIME_SPEC_H
#define UTILITIES_TIME_TIME_SPEC_H

#include "Chrono.h"

#include <ctime>
#include <ostream>
#include <type_traits>

namespace Utilities
{
namespace Time
{

//------------------------------------------------------------------------------
/// \brief ::timespec struct wrapper using inheritance.
/// struct timespec
/// {
///   time_t tv_sec; // Seconds
///   long tv_nsec; // Nanoseconds
/// };
//------------------------------------------------------------------------------
struct TimeSpec : public ::timespec
{
  using ::timespec::timespec;

  TimeSpec();

  explicit TimeSpec(
    const long time_value_sec,
    const long time_value_nsec = 0);

  template <
    class Duration,
    typename std::enable_if_t<is_duration<Duration>::value, int> = 0
    >
  explicit TimeSpec(const Duration& duration)
  {
    const Seconds duration_secs {duration_cast<Seconds>(duration)};
    const Nanoseconds duration_nanosecs {
      duration_cast<Nanoseconds>(duration - duration_secs)};

    tv_sec = duration_secs.count();
    tv_nsec = duration_nanosecs.count();
  }

  virtual ~TimeSpec() = default;

  long get_seconds() const
  {
    return tv_sec;
  }

  long get_nanoseconds() const
  {
    return tv_nsec;
  }

  // cf. https://stackoverflow.com/questions/7409565/how-to-use-reinterpret-cast-to-cast-to-a-derived-class-pointer-in-c

  const ::timespec* to_timespec() const
  {
    return static_cast<const ::timespec*>(this);
  }

  ::timespec* to_timespec()
  {
    return static_cast<::timespec*>(this);
  }

  friend std::ostream& operator<<(
    std::ostream& os,
    const TimeSpec& time_specification);
};

TimeSpec operator-(const TimeSpec& t1, const TimeSpec& t2);

} // namespace Time
} // namespace Utilities

#endif // UTILITIES_TIME_TIME_SPEC_H