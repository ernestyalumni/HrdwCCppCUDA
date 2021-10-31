#include "TimeSpecification.h"

#include <ctime> // ::timespec
#include <ostream>

using std::ostream;

namespace Utilities
{

namespace Time
{

TimeSpecification::TimeSpecification():
  timespec_{0, 0}
{}

TimeSpecification::TimeSpecification(const ::timespec& timespec)
  timespec_{timespec}
{}

ostream& TimeSpecification::operator<<(
  ostream& os,
  const TimeSpecification& time_specification)
{
  os << time_specification.timespec_.tv_sec <<
    ' ' <<
    time_specification.timespec_.tv_nsec <<
    "\n";

  return os;
}

} // namespace Time
} // namespace Utilities