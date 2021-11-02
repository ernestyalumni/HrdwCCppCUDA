#include "Utilities/Time/ClockId.h"
#include "Utilities/Time/GetClockTime.h"
#include "Utilities/Time/TimeSpecification.h"

#include <ctime>
#include <gtest/gtest.h>

#include <iostream>

using Utilities::Time::ClockId;
using Utilities::Time::Details::carry_or_borrow_nanoseconds;
using Utilities::Time::GetClockTime;
using Utilities::Time::GetMonotonicClockTime;
using Utilities::Time::GetRealTimeClockTime;
using Utilities::Time::TimeSpecification;

namespace GoogleUnitTests
{
namespace Utilities
{
namespace Time
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetClockTimeTests, ConstructsWithClockId)
{
  {
    GetClockTime gct {ClockId::real_time};
  }
  {
    GetClockTime gct {ClockId::monotonic};
  }
  {
    GetClockTime gct {ClockId::process_cpu_time};
  }
  {
    GetClockTime gct {ClockId::thread_cpu_time};
  }

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetClockTimeTests, RealTimeClockIsMonotonic)
{
  GetClockTime gct {ClockId::real_time};

  const TimeSpecification t1 {gct()};

  TimeSpecification sleep_for {0, 1000};
  ::nanosleep(sleep_for.to_timespec_pointer(), nullptr);

  const TimeSpecification t2 {gct()};

  EXPECT_TRUE(t2 >= t1);

  const TimeSpecification difference {t2 - t1};

  EXPECT_TRUE(difference.get_nanoseconds() >= 1000);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetClockTimeTests, MonotonicClockIsMonotonic)
{
  GetClockTime gct {ClockId::monotonic};

  const TimeSpecification t1 {gct()};

  TimeSpecification sleep_for {0, 1000};
  ::nanosleep(sleep_for.to_timespec_pointer(), nullptr);

  const TimeSpecification t2 {gct()};

  EXPECT_TRUE(t2 >= t1);

  const TimeSpecification difference {
    carry_or_borrow_nanoseconds(t2) - carry_or_borrow_nanoseconds(t1)};

  EXPECT_TRUE(difference.get_seconds() >= 0);
  EXPECT_TRUE(difference.get_nanoseconds() >= 1000);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetRealTimeClockTimeTests, ClockIsMonotonic)
{
  GetRealTimeClockTime grct {};

  const TimeSpecification t1 {grct()};

  TimeSpecification sleep_for {0, 10000};
  ::nanosleep(sleep_for.to_timespec_pointer(), nullptr);

  const TimeSpecification t2 {grct()};

  EXPECT_TRUE(t2 >= t1);

  const TimeSpecification difference {t2 - t1};

  EXPECT_TRUE(difference.get_nanoseconds() >= 10000);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetMonotonicClockTimeTests, ClockIsMonotonic)
{
  GetMonotonicClockTime gmct {};

  const TimeSpecification t1 {gmct()};

  TimeSpecification sleep_for {0, 10000};
  ::nanosleep(sleep_for.to_timespec_pointer(), nullptr);

  const TimeSpecification t2 {gmct()};

  EXPECT_TRUE(t2 >= t1);

  const TimeSpecification difference {t2 - t1};

  EXPECT_TRUE(difference.get_nanoseconds() >= 10000);
}

} // namespace Time
} // namespace Utilities
} // namespace GoogleUnitTests
