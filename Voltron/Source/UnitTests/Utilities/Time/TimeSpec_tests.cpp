#include "Utilities/Time/TimeSpec.h"

#include "Utilities/Time/Chrono.h"

#include <boost/test/unit_test.hpp>
#include <utility>

using Utilities::Time::TimeSpec;

BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(Time)
BOOST_AUTO_TEST_SUITE(TimeSpec_tests)

TimeSpec create_rvalue_timespec(const long s, const long ns)
{
  TimeSpec ts {s, ns};
  return ts;
}

TimeSpec create_copy_elision_timespec(const long s, const long ns)
{
  return TimeSpec{s, ns};
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
  TimeSpec ts {};
  BOOST_TEST(ts.tv_sec == 0);
  BOOST_TEST(ts.tv_nsec == 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CopyConstructionCopies)
{
  TimeSpec ts1 {42, 69};
  TimeSpec ts2 {ts1};
  BOOST_TEST(ts1.tv_sec == 42);
  BOOST_TEST(ts1.tv_nsec == 69);
  BOOST_TEST(ts2.tv_sec == 42);
  BOOST_TEST(ts2.tv_nsec == 69);

  // Modify the source TimeSpec to show target is not affected.
  ts1.tv_sec = 19;
  ts1.tv_nsec = 89;

  BOOST_TEST(ts1.tv_sec == 19);
  BOOST_TEST(ts1.tv_nsec == 89);
  BOOST_TEST(ts2.tv_sec == 42);
  BOOST_TEST(ts2.tv_nsec == 69);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CopyAssignmentCopies)
{
  TimeSpec ts1 {42, 69};
  TimeSpec ts2 {};
  BOOST_TEST(ts2.tv_sec == 0);
  BOOST_TEST(ts2.tv_nsec == 0);

  ts2 = ts1;

  BOOST_TEST(ts1.tv_sec == 42);
  BOOST_TEST(ts1.tv_nsec == 69);
  BOOST_TEST(ts2.tv_sec == 42);
  BOOST_TEST(ts2.tv_nsec == 69);

  // Modify the source TimeSpec to show target is not affected.
  ts1.tv_sec = 19;
  ts1.tv_nsec = 89;

  BOOST_TEST(ts1.tv_sec == 19);
  BOOST_TEST(ts1.tv_nsec == 89);
  BOOST_TEST(ts2.tv_sec == 42);
  BOOST_TEST(ts2.tv_nsec == 69);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MoveConstructionCopiesRvalues)
{
  TimeSpec ts1 {42, 69};
  TimeSpec ts2 {std::move(ts1)};
  BOOST_TEST(ts1.tv_sec == 42);
  BOOST_TEST(ts1.tv_nsec == 69);
  BOOST_TEST(ts2.tv_sec == 42);
  BOOST_TEST(ts2.tv_nsec == 69);

  // Modify the source TimeSpec to show target is not affected.
  ts1.tv_sec = 19;
  ts1.tv_nsec = 89;

  BOOST_TEST(ts1.tv_sec == 19);
  BOOST_TEST(ts1.tv_nsec == 89);
  BOOST_TEST(ts2.tv_sec == 42);
  BOOST_TEST(ts2.tv_nsec == 69);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MoveConstructionFromFunctionReturn)
{
  {
    TimeSpec ts {create_rvalue_timespec(42, 69)};
    BOOST_TEST(ts.tv_sec == 42);
    BOOST_TEST(ts.tv_nsec == 69);
  }
  {
    TimeSpec ts {create_copy_elision_timespec(42, 69)};
    BOOST_TEST(ts.tv_sec == 42);
    BOOST_TEST(ts.tv_nsec == 69);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MoveAssignmentCopiesRvalues)
{
  TimeSpec ts1 {42, 69};
  TimeSpec ts2 {};
  BOOST_TEST(ts2.tv_sec == 0);
  BOOST_TEST(ts2.tv_nsec == 0);

  ts2 = std::move(ts1);
  BOOST_TEST(ts1.tv_sec == 42);
  BOOST_TEST(ts1.tv_nsec == 69);
  BOOST_TEST(ts2.tv_sec == 42);
  BOOST_TEST(ts2.tv_nsec == 69);

  // Modify the source TimeSpec to show target is not affected.
  ts1.tv_sec = 19;
  ts1.tv_nsec = 89;

  BOOST_TEST(ts1.tv_sec == 19);
  BOOST_TEST(ts1.tv_nsec == 89);
  BOOST_TEST(ts2.tv_sec == 42);
  BOOST_TEST(ts2.tv_nsec == 69);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MoveAssignmentFromFunctionReturn)
{
  {
    TimeSpec ts {};
    BOOST_TEST(ts.tv_sec == 0);
    BOOST_TEST(ts.tv_nsec == 0);

    ts = create_rvalue_timespec(42, 69);
    BOOST_TEST(ts.tv_sec == 42);
    BOOST_TEST(ts.tv_nsec == 69);
  }
  {
    TimeSpec ts {};
    BOOST_TEST(ts.tv_sec == 0);
    BOOST_TEST(ts.tv_nsec == 0);

    ts = create_copy_elision_timespec(42, 69);
    BOOST_TEST(ts.tv_sec == 42);
    BOOST_TEST(ts.tv_nsec == 69);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ToTimeSpecPointsToData)
{
  TimeSpec ts {42, 69};

  ::timespec* timespec_ptr {nullptr};

  timespec_ptr = ts.to_timespec();

  BOOST_TEST(timespec_ptr->tv_sec == 42);
  BOOST_TEST(timespec_ptr->tv_nsec == 69);

  TimeSpec another_ts {-69, -42};

  timespec_ptr = another_ts.to_timespec();

  BOOST_TEST(timespec_ptr->tv_sec == -69);
  BOOST_TEST(timespec_ptr->tv_nsec == -42);

  another_ts.tv_sec = 19;
  another_ts.tv_nsec = 89;

  BOOST_TEST(timespec_ptr->tv_sec == 19);
  BOOST_TEST(timespec_ptr->tv_nsec == 89);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SubtractionWorksForNanosecondRHSSmallerThanLHS)
{
  {
    const TimeSpec lhs {42, 69};
    const TimeSpec rhs {42, 68};
    const TimeSpec difference {lhs - rhs};

    BOOST_TEST(difference.tv_sec == 0);
    BOOST_TEST(difference.tv_nsec == 1);
  }
  {
    const TimeSpec lhs {43, 999'999'999};
    const TimeSpec rhs {42, 0};
    const TimeSpec difference {lhs - rhs};

    BOOST_TEST(difference.tv_sec == 1);
    BOOST_TEST(difference.tv_nsec == 999'999'999);
  }
  {
    const TimeSpec lhs {-43, 999'999'999};
    const TimeSpec rhs {42, 999'999'998};
    const TimeSpec difference {lhs - rhs};

    BOOST_TEST(difference.tv_sec == -85);
    BOOST_TEST(difference.tv_nsec == 1);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SubtractionBorrowsWhenNanosecondsLHSLessThanRHS)
{
  {
    const TimeSpec lhs {42, 69};
    const TimeSpec rhs {42, 70};
    const TimeSpec difference {lhs - rhs};

    BOOST_TEST(difference.tv_sec == -1);
    BOOST_TEST(difference.tv_nsec == 999'999'999);
  }
  {
    const TimeSpec lhs {43, 0};
    const TimeSpec rhs {42, 999'999'999};
    const TimeSpec difference {lhs - rhs};

    BOOST_TEST(difference.tv_sec == 0);
    BOOST_TEST(difference.tv_nsec == 1);
  }
  {
    const TimeSpec lhs {-43, 999'999'998};
    const TimeSpec rhs {42, 999'999'999};
    const TimeSpec difference {lhs - rhs};

    BOOST_TEST(difference.tv_sec == -86);
    BOOST_TEST(difference.tv_nsec == 999'999'999);
  }
}

BOOST_AUTO_TEST_SUITE_END() // TimeSpec_tests
BOOST_AUTO_TEST_SUITE_END() // Time
BOOST_AUTO_TEST_SUITE_END() // Utilities