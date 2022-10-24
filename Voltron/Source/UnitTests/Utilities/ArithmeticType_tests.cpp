#include "Utilities/ArithmeticType.h"

#include <boost/test/unit_test.hpp>

using Utilities::Kedyk::ArithmeticType;

BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(ArithmeticType_tests)

struct TestIntType
{
  TestIntType(const int a = 0):
    data_{a}
  {}

  int data_;
};

class ArithmeticInt : public ArithmeticType<TestIntType>
{
  public:
    ArithmeticInt(const TestIntType a = TestIntType{0}):
      data_{a}
    {}

    TestIntType data_;
};

//------------------------------------------------------------------------------
/// \details It was an interesting fact that the friend operators, e.g. friend T
/// operator+, does not compile and requires a class or enumeration type for the
/// template parameter of ArithmeticType.
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithStructType)
{
  TestIntType a {2};
  TestIntType b {3};
  ArithmeticInt aa {a};
  ArithmeticInt bb {b};
}

BOOST_AUTO_TEST_SUITE_END() // ArithmeticType_tests
BOOST_AUTO_TEST_SUITE_END() // Utilities