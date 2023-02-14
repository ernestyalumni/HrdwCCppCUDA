#include "Algorithms/Sorting/Strings/MostSignificantDigitFirst.h"
#include "TestValues.h"

#include <boost/test/unit_test.hpp>

using namespace Algorithms::Sorting::Strings::MostSignificantDigitFirst;
using UnitTests::Algorithms::Sorting::Strings::
  create_license_plates_example_input;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(Sorting)
BOOST_AUTO_TEST_SUITE(Strings)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IsLessComparesDifferentLengthStrings)
{
  BOOST_TEST(is_less("better", "have", 0));
  BOOST_TEST(!is_less("better", "have", 1));
  BOOST_TEST(!is_less("mymoney", "mymoney", 0));
  BOOST_TEST(!is_less("mymoney", "mymoney", 1));
  BOOST_TEST(is_less("betterhavemymone", "betterhavemymoney", 0));
  BOOST_TEST(is_less("betterhavemymone", "betterhavemymoney", 2));
}

BOOST_AUTO_TEST_SUITE_END() // Strings
BOOST_AUTO_TEST_SUITE_END() // Sorting
BOOST_AUTO_TEST_SUITE_END() // Algorithms