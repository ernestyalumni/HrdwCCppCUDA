#ifndef UTILITIES_TIME_GET_CLOCK_TIME_H
#define UTILITIES_TIME_GET_CLOCK_TIME_H

#include "ClockId.h"
#include "TimeSpecification.h"

#include <optional>

namespace Utilities
{

namespace Time
{

class GetClockTime
{
  public:

    GetClockTime() = delete;

    explicit GetClockTime(const ClockId clock_id);

    TimeSpecification operator()() const;

    std::optional<TimeSpecification> get_clock_time() const;

  private:

    ClockId clock_id_;
};

class GetMonotonicClockTime : public GetClockTime
{
  public:

    GetMonotonicClockTime();
};

class GetRealTimeClockTime : public GetClockTime
{
  public:

    GetRealTimeClockTime();
};

} // namespace Time
} // namespace Utilities

#endif // UTILITIES_TIME_GET_CLOCK_TIME_H