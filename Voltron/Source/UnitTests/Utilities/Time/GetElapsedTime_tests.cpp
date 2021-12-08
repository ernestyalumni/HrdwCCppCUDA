#include "Utilities/Time/GetElapsedTime.h"

#include "Utilities/Time/ClockId.h"
#include "Utilities/Time/TimeSpec.h"

#include <boost/test/unit_test.hpp>
#include <ctime>

using Utilities::Time::ClockId;
using Utilities::Time::GetElapsedTime;
using Utilities::Time::TimeSpec;

BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(Time)
BOOST_AUTO_TEST_SUITE(GetElapsedTime_tests)


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FunctionCallOperatorGetsElapsedTime)
{
  TimeSpec sleep_for {0, 10000};

  GetElapsedTime<> elapsed_time {};

  elapsed_time.start();

  ::nanosleep(sleep_for.to_timespec(), nullptr);

  const TimeSpec difference {elapsed_time()};

  BOOST_TEST(difference.tv_nsec >= 10000);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FunctionCallOperatorGetsElapsedTimeForRealTimeClock)
{
  TimeSpec sleep_for {0, 10000};

  GetElapsedTime<ClockId::real_time> elapsed_time {};

  elapsed_time.start();

  ::nanosleep(sleep_for.to_timespec(), nullptr);

  const TimeSpec difference {elapsed_time()};

  BOOST_TEST(difference.tv_nsec >= 10000);
}

BOOST_AUTO_TEST_SUITE_END() // GetElapsedTime_tests
BOOST_AUTO_TEST_SUITE_END() // Time
BOOST_AUTO_TEST_SUITE_END() // Utilities