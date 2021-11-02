#include "Utilities/Time/TimeSpecification.h"

#include <ctime>
#include <gtest/gtest.h>
#include <limits>
#include <sstream>
#include <type_traits>

using Utilities::Time::Details::carry_or_borrow_nanoseconds;
using Utilities::Time::TimeSpecification;
using std::is_signed;
using std::numeric_limits;

class TestTimeSpecification : public TimeSpecification
{
  public:

    using TimeSpecification::TimeSpecification;
    using TimeSpecification::size_of_tv_sec;
    using TimeSpecification::size_of_tv_nsec;
    using TimeSpecification::alignment_of_tv_sec;
    using TimeSpecification::alignment_of_tv_nsec;
    using TimeSpecification::alignment_of_this;
};

namespace GoogleUnitTests
{
namespace Utilities
{
namespace Time
{

namespace Details
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CarryOrBorrowNanosecondsTests, ReturnsSameValueForNonNegativeNanoseconds)
{
  {
    const ::timespec ts {42, 69};
    const ::timespec result {carry_or_borrow_nanoseconds(ts)};
    EXPECT_EQ(result.tv_sec, 42);
    EXPECT_EQ(result.tv_nsec, 69);
  }
  {
    const ::timespec ts {42, 999'999'999};
    const ::timespec result {carry_or_borrow_nanoseconds(ts)};
    EXPECT_EQ(result.tv_sec, 42);
    EXPECT_EQ(result.tv_nsec, 999'999'999);
  }
  {
    const ::timespec ts {-42, 999'999'999};
    const ::timespec result {carry_or_borrow_nanoseconds(ts)};
    EXPECT_EQ(result.tv_sec, -42);
    EXPECT_EQ(result.tv_nsec, 999'999'999);
  }
  {
    const ::timespec ts {-69, 0};
    const ::timespec result {carry_or_borrow_nanoseconds(ts)};
    EXPECT_EQ(result.tv_sec, -69);
    EXPECT_EQ(result.tv_nsec, 0);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CarryOrBorrowNanosecondsTests, CarriesForLargeNonNegativeNanoseconds)
{
  {
    const ::timespec ts {42, 1'000'000'000};
    const ::timespec result {carry_or_borrow_nanoseconds(ts)};
    EXPECT_EQ(result.tv_sec, 43);
    EXPECT_EQ(result.tv_nsec, 0);
  }
  {
    const ::timespec ts {43, 2'345'678'901};
    const ::timespec result {carry_or_borrow_nanoseconds(ts)};
    EXPECT_EQ(result.tv_sec, 45);
    EXPECT_EQ(result.tv_nsec, 345'678'901);
  }
  {
    const ::timespec ts {-42, 43'999'999'999};
    const ::timespec result {carry_or_borrow_nanoseconds(ts)};
    EXPECT_EQ(result.tv_sec, 1);
    EXPECT_EQ(result.tv_nsec, 999'999'999);
  }
  {
    const ::timespec ts {-69, 1'000'000'000};
    const ::timespec result {carry_or_borrow_nanoseconds(ts)};
    EXPECT_EQ(result.tv_sec, -68);
    EXPECT_EQ(result.tv_nsec, 0);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CarryOrBorrowNanosecondsTests, BorrowsForNegativeNanoseconds)
{
  {
    const ::timespec ts {42, -1};
    const ::timespec result {carry_or_borrow_nanoseconds(ts)};
    EXPECT_EQ(result.tv_sec, 41);
    EXPECT_EQ(result.tv_nsec, 999'999'999);
  }
  {
    const ::timespec ts {43, -2'345'678'901};
    const ::timespec result {carry_or_borrow_nanoseconds(ts)};
    EXPECT_EQ(result.tv_sec, 40);
    EXPECT_EQ(result.tv_nsec, 654'321'099);
  }
  {
    const ::timespec ts {-42, -43'999'999'999};
    const ::timespec result {carry_or_borrow_nanoseconds(ts)};
    EXPECT_EQ(result.tv_sec, -86);
    EXPECT_EQ(result.tv_nsec, 1);
  }
  {
    const ::timespec ts {-69, -1'000'000'000};
    const ::timespec result {carry_or_borrow_nanoseconds(ts)};
    EXPECT_EQ(result.tv_sec, -70);
    EXPECT_EQ(result.tv_nsec, 0);
  }
}

} // namespace Details

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TimeSpecificationTests, DefaultConstructs)
{
  const TimeSpecification ts;

  EXPECT_EQ(ts.get_timespec().tv_sec, 0);
  EXPECT_EQ(ts.get_timespec().tv_nsec, 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TimeSpecificationTests, SizeOfTvSec)
{
  const TestTimeSpecification ts;

  EXPECT_EQ(ts.size_of_tv_sec(), 8);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TimeSpecificationTests, SizeOfTvNSec)
{
  const TestTimeSpecification ts;

  EXPECT_EQ(ts.size_of_tv_nsec(), 8);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TimeSpecificationTests, AlignmentOfTvSec)
{
  const TestTimeSpecification ts;

  EXPECT_EQ(ts.alignment_of_tv_sec(), 8);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TimeSpecificationTests, AlignmentOfTvNSec)
{
  const TestTimeSpecification ts;

  EXPECT_EQ(ts.alignment_of_tv_nsec(), 8);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TimeSpecificationTests, AlignmentOfThis)
{
  const TestTimeSpecification ts;

  EXPECT_EQ(ts.alignment_of_this(), 8);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TimeSpecificationTests, UnderlyingTypesAreSigned)
{
  const TimeSpecification ts;

  EXPECT_TRUE(is_signed<decltype(ts.get_timespec().tv_sec)>::value);
  EXPECT_TRUE(is_signed<decltype(ts.get_timespec().tv_nsec)>::value);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TimeSpecificationTests, NumericLimitsAreSignedAndLong)
{
  const TimeSpecification ts;

  // -9223372036854775808
  EXPECT_EQ(
    numeric_limits<decltype(ts.get_timespec().tv_sec)>::min(),
    numeric_limits<long>::min());
  // 9223372036854775807
  EXPECT_EQ(
    numeric_limits<decltype(ts.get_timespec().tv_sec)>::max(),
    numeric_limits<long>::max());
  // -9223372036854775808
  EXPECT_EQ(
    numeric_limits<decltype(ts.get_timespec().tv_sec)>::lowest(),
    numeric_limits<long>::lowest());

  // -9223372036854775808
  EXPECT_EQ(
    numeric_limits<decltype(ts.get_timespec().tv_nsec)>::min(),
    numeric_limits<long>::min());
  // 9223372036854775807
  EXPECT_EQ(
    numeric_limits<decltype(ts.get_timespec().tv_nsec)>::max(),
    numeric_limits<long>::max());
  // -9223372036854775808
  EXPECT_EQ(
    numeric_limits<decltype(ts.get_timespec().tv_nsec)>::lowest(),
    numeric_limits<long>::lowest());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TimeSpecificationTests, ToTimespecPointerPointsToEncapsulatedObject)
{
  TimeSpecification ts {42, 69};

  ::timespec* timespec_ptr {nullptr};

  timespec_ptr = ts.to_timespec_pointer();

  EXPECT_EQ(timespec_ptr->tv_sec, 42);
  EXPECT_EQ(timespec_ptr->tv_nsec, 69);

  TimeSpecification another_ts {-69, -42};

  timespec_ptr = another_ts.to_timespec_pointer();

  EXPECT_EQ(timespec_ptr->tv_sec, -69);
  EXPECT_EQ(timespec_ptr->tv_nsec, -42);  
}

// This test demonstrated that doing a reinterpret_cast<::timespec*> on this
// class that uses composition on ::timespec fails to align data members since
// the object's first bytes are for a pointer address.
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// TEST(TimeSpecificationTests, AsTimespecPointerFailsToPointToEncapsulatedObject)
//{
//  TimeSpecification ts {42, 69};

//  ::timespec* timespec_ptr {nullptr};

//  timespec_ptr = ts.as_timespec_pointer();

//  EXPECT_TRUE(timespec_ptr->tv_sec != 42);
//  EXPECT_EQ(timespec_ptr->tv_nsec, 42);

//  TimeSpecification another_ts {-69, -42};

//  timespec_ptr = another_ts.as_timespec_pointer();

//  EXPECT_TRUE(timespec_ptr->tv_sec != -69);
//  EXPECT_EQ(timespec_ptr->tv_nsec, -69);
//}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TimeSpecificationTests, GreaterThanOrEqualForNormalizedTimeSpecifications)
{
  {
    const TimeSpecification lhs {42, 69};
    const TimeSpecification rhs {42, 70};
    EXPECT_FALSE(lhs >= rhs);
  }
  {
    const TimeSpecification lhs {43, 0};
    const TimeSpecification rhs {42, 999'999'999};
    EXPECT_TRUE(lhs >= rhs);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TimeSpecificationTests, SubtractionWorksForNanosecondRHSSmallerThanLHS)
{
  {
    const TimeSpecification lhs {42, 69};
    const TimeSpecification rhs {42, 68};
    const TimeSpecification difference {lhs - rhs};

    EXPECT_EQ(difference.get_timespec().tv_sec, 0);
    EXPECT_EQ(difference.get_timespec().tv_nsec, 1);
  }
  {
    const TimeSpecification lhs {43, 999'999'999};
    const TimeSpecification rhs {42, 0};
    const TimeSpecification difference {lhs - rhs};

    EXPECT_EQ(difference.get_timespec().tv_sec, 1);
    EXPECT_EQ(difference.get_timespec().tv_nsec, 999'999'999);
  }
  {
    const TimeSpecification lhs {-43, 999'999'999};
    const TimeSpecification rhs {42, 999'999'998};
    const TimeSpecification difference {lhs - rhs};

    EXPECT_EQ(difference.get_timespec().tv_sec, -85);
    EXPECT_EQ(difference.get_timespec().tv_nsec, 1);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TimeSpecificationTests, SubtractionBorrowsWhenNanosecondsLHSLessThanRHS)
{
  {
    const TimeSpecification lhs {42, 69};
    const TimeSpecification rhs {42, 70};
    const TimeSpecification difference {lhs - rhs};

    EXPECT_EQ(difference.get_timespec().tv_sec, -1);
    EXPECT_EQ(difference.get_timespec().tv_nsec, 999'999'999);
  }
  {
    const TimeSpecification lhs {43, 0};
    const TimeSpecification rhs {42, 999'999'999};
    const TimeSpecification difference {lhs - rhs};

    EXPECT_EQ(difference.get_timespec().tv_sec, 0);
    EXPECT_EQ(difference.get_timespec().tv_nsec, 1);
  }
  {
    const TimeSpecification lhs {-43, 999'999'998};
    const TimeSpecification rhs {42, 999'999'999};
    const TimeSpecification difference {lhs - rhs};

    EXPECT_EQ(difference.get_timespec().tv_sec, -86);
    EXPECT_EQ(difference.get_timespec().tv_nsec, 999'999'999);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TimeSpecificationTests, OstreamReturnsZeroValues)
{
  const TimeSpecification ts;

  std::ostringstream ss;

  ss << ts;

  EXPECT_EQ(ss.str(), "0 0\n");
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TimeSpecificationTests, OstreamReturnsNonZeroValues)
{
  const TimeSpecification ts {42, 69};

  std::ostringstream ss;

  ss << ts;

  EXPECT_EQ(ss.str(), "42 69\n");
}

} // namespace Time
} // namespace Utilities
} // namespace GoogleUnitTests
