#include "Utilities/Time/TimeSpecification.h"

#include <gtest/gtest.h>

using Utilities::Time::TimeSpecification;

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TimeSpecificationTests, DefaultConstructs)
{
  const TimeSpecification ts;

  EXPECT_EQ(ts.tv_sec, 0);
  EXPECT_EQ(ts.tv_nsec, 0);
}