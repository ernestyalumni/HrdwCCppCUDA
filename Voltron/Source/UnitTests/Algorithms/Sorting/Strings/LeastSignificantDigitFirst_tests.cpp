#include "Algorithms/Sorting/Strings/LeastSignificantDigitFirst.h"
#include "DataStructures/Arrays/DynamicArray.h"

#include <boost/test/unit_test.hpp>
#include <string>

using Algorithms::Sorting::Strings::least_significant_digit_first_sort;

template <typename T>
using Array = DataStructures::Arrays::PrimitiveDynamicArray<T>;

using std::string;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(Sorting)
BOOST_AUTO_TEST_SUITE(Strings)

Array<string> create_license_plates_example_input()
{
  Array<string> license_plates {};

  license_plates.initialize({
    "4PGC938",
    "2IYE230",
    "3CIO720",
    "1ICK750",
    "1OHV845",
    "4JZY524",
    "1ICK750",
    "3CIO720",
    "1OHV845",
    "1OHV845",
    "2RLA629",
    "2RLA629",
    "3ATW723"});

  BOOST_TEST_REQUIRE(license_plates.size() == 13);
  BOOST_TEST_REQUIRE(license_plates[0] == "4PGC938");
  BOOST_TEST_REQUIRE(license_plates[1] == "2IYE230");
  BOOST_TEST_REQUIRE(license_plates[12] == "3ATW723");

  return std::move(license_plates);
}

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