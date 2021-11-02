#include "GetClockTime.h"

#include "ClockId.h"
#include "TimeSpecification.h"
#include "Utilities/TypeSupport/GetUnderlyingValue.h"

#include <ctime>
#include <optional>

using Utilities::TypeSupport::get_underlying_value;

namespace Utilities
{

namespace Time
{

GetClockTime::GetClockTime(const ClockId clock_id):
  clock_id_{clock_id}
{}

TimeSpecification GetClockTime::operator()() const
{
  TimeSpecification time_specification {};

  ::clock_gettime(
    get_underlying_value<ClockId>(clock_id_),
    time_specification.to_timespec_pointer());

  return time_specification;
}

std::optional<TimeSpecification> GetClockTime::get_clock_time() const
{
  TimeSpecification time_specification {};

  int return_value {::clock_gettime(
    get_underlying_value<ClockId>(clock_id_),
    time_specification.to_timespec_pointer())};


  return return_value < 0 ?
    std::nullopt :
      std::make_optional<TimeSpecification>(time_specification);
}

GetMonotonicClockTime::GetMonotonicClockTime():
  GetClockTime{ClockId::monotonic}
{}

GetRealTimeClockTime::GetRealTimeClockTime():
  GetClockTime{ClockId::real_time}
{}

} // namespace Time
} // namespace Utilities
