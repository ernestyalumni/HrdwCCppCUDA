#ifndef UTILITIES_TIME_GET_ELAPSED_TIME_H
#define UTILITIES_TIME_GET_ELAPSED_TIME_H

#include "ClockId.h"
#include "GetClockTime.h"
#include "TimeSpec.h"

#include <ctime>

namespace Utilities
{
namespace Time
{

template <ClockId ClockIdT = ClockId::monotonic>
class GetElapsedTime
{
  public:

    GetElapsedTime():
      t_0_{}
    {
      start();
    }

    void start()
    {
      t_0_ = get_clock_time<ClockIdT>();
    }

    TimeSpec operator()()
    {
      return get_clock_time<ClockIdT>() - t_0_;
    }

  private:

    TimeSpec t_0_;
};

} // namespace Time
} // namespace Utilities

#endif // UTILITIES_TIME_GET_ELAPSED_TIME_H