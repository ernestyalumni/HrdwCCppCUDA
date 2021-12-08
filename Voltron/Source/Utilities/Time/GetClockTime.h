#ifndef UTILITIES_TIME_GET_CLOCK_TIME_H
#define UTILITIES_TIME_GET_CLOCK_TIME_H

#include "ClockId.h"
#include "TimeSpec.h"
#include "Cpp/Utilities/TypeSupport/UnderlyingTypes.h"

#include <ctime>

namespace Utilities
{
namespace Time
{

template <ClockId ClockIdT = ClockId::monotonic>
TimeSpec get_clock_time()
{
  TimeSpec ts {};

  ::clock_gettime(
    Cpp::Utilities::TypeSupport::get_underlying_value<ClockId>(ClockIdT),
    ts.to_timespec());

  return ts;
}

} // namespace Time
} // namespace Utilities

#endif // UTILITIES_TIME_GET_CLOCK_TIME_H