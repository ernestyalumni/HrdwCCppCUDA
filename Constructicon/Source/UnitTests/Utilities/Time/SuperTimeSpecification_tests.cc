#include "Utilities/Time/SuperTimeSpecification.h"

#include <gtest/gtest.h>

using Utilities::Time::SuperTimeSpecification;

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SuperTimeSpecificationTests, DefaultConstructs)
{
  const SuperTimeSpecification ts;

  EXPECT_EQ(ts.tv_sec, 0);
  EXPECT_EQ(ts.tv_nsec, 0);
}