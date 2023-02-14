#include "Algorithms/Sorting/Strings/LeastSignificantDigitFirst.h"
#include "TestValues.h"

#include <boost/test/unit_test.hpp>

using Algorithms::Sorting::Strings::least_significant_digit_first_sort;
using UnitTests::Algorithms::Sorting::Strings::
  create_license_plates_example_input;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(Sorting)
BOOST_AUTO_TEST_SUITE(Strings)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(LeastSignificantDigitFirstSortSorts)
{ 
  auto license_plates {create_license_plates_example_input()};

  least_significant_digit_first_sort(license_plates, 7);

  BOOST_TEST(license_plates[0] == "1ICK750");
  BOOST_TEST(license_plates[1] == "1ICK750");
  BOOST_TEST(license_plates[2] == "1OHV845");
  BOOST_TEST(license_plates[3] == "1OHV845");
  BOOST_TEST(license_plates[11] == "4JZY524");
  BOOST_TEST(license_plates[12] == "4PGC938");
}

BOOST_AUTO_TEST_SUITE_END() // Strings
BOOST_AUTO_TEST_SUITE_END() // Sorting
BOOST_AUTO_TEST_SUITE_END() // Algorithms