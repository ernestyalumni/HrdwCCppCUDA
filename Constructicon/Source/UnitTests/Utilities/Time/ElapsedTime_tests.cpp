#include "Utilities/Time/ElapsedTime.h"
#include "Utilities/Time/TimeSpecification.h"

#include <ctime>
#include <gtest/gtest.h>

using Utilities::Time::MonotonicElapsedTime;
using Utilities::Time::RealTimeElapsedTime;
using Utilities::Time::TimeSpecification;

namespace GoogleUnitTests
{
namespace Utilities
{
namespace Time
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(MonotonicElapsedTimeTests, DefaultConstructs)
{
  MonotonicElapsedTime elapsed_time;

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(RealTimeElapsedTimeTests, DefaultConstructs)
{
  RealTimeElapsedTime elapsed_time;

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(MonotonicElapsedTimeTests, StartsUponConstruction)
{
  TimeSpecification sleep_for {0, 10000};

  MonotonicElapsedTime elapsed_time;

  ::nanosleep(sleep_for.to_timespec_pointer(), nullptr);

  const TimeSpecification difference {elapsed_time()};

  EXPECT_TRUE(difference.get_nanoseconds() >= 10000);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(RealTimeElapsedTimeTests, StartsUponConstruction)
{
  TimeSpecification sleep_for {0, 10000};

  RealTimeElapsedTime elapsed_time;

  ::nanosleep(sleep_for.to_timespec_pointer(), nullptr);

  const TimeSpecification difference {elapsed_time()};

  EXPECT_TRUE(difference.get_nanoseconds() >= 10000);
}

} // namespace Time
} // namespace Utilities
} // namespace GoogleUnitTests