#include "Algorithms/Sorting/Strings/MostSignificantDigitFirst.h"
#include "DataStructures/Arrays/DynamicArray.h"
#include "TestValues.h"

#include <boost/test/unit_test.hpp>
#include <utility>

template <typename T>
using Array = DataStructures::Arrays::PrimitiveDynamicArray<T>;

using namespace Algorithms::Sorting::Strings::MostSignificantDigitFirst;
using UnitTests::Algorithms::Sorting::Strings::
  create_license_plates_example_input;

using std::string;

Array<string> create_different_length_strings()
{
  Array<string> words {};

  words.initialize({
    "she",
    "sells",
    "seashells",
    "by",
    "the",
    "seashore",
    "the",
    "shells",
    "she",
    "sells",
    "are",
    "surely",
    "seashells"});

  BOOST_TEST_REQUIRE(words.size() == 13);
  BOOST_TEST_REQUIRE(words[0] == "she");
  BOOST_TEST_REQUIRE(words[1] == "sells");
  BOOST_TEST_REQUIRE(words[12] == "seashells");

  return std::move(words);
}

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(Sorting)
BOOST_AUTO_TEST_SUITE(Strings)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IsLessComparesDifferentLengthStrings)
{
  BOOST_TEST(is_less("better", "have", 0));
  BOOST_TEST(!is_less("better", "have", 1));
  BOOST_TEST(is_less("betterhavemymone", "betterhavemymoney", 0));
  BOOST_TEST(is_less("betterhavemymone", "betterhavemymoney", 2));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IsLessReturnsFalseForSameStrings)
{
  BOOST_TEST(!is_less("mymoney", "mymoney", 0));
  BOOST_TEST(!is_less("mymoney", "mymoney", 1));
  BOOST_TEST(!is_less("mymoney", "mymoney", 2));
  BOOST_TEST(!is_less("mymoney", "mymoney", 3));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(InsertionSortWorksOnSubarrays)
{
  Array<string> words {create_different_length_strings()};

  insertion_sort(words, 0, 6, 0);

  BOOST_TEST(words[0] == "by");
  BOOST_TEST(words[1] == "seashells");
  BOOST_TEST(words[2] == "seashore");
  BOOST_TEST(words[3] == "sells");
  BOOST_TEST(words[4] == "she");
  BOOST_TEST(words[5] == "the");
  BOOST_TEST(words[6] == "the");

  insertion_sort(words, 6, 9, 0);

  BOOST_TEST(words[0] == "by");
  BOOST_TEST(words[1] == "seashells");

  BOOST_TEST(words[6] == "sells");
  BOOST_TEST(words[7] == "she");
  BOOST_TEST(words[8] == "shells");
  BOOST_TEST(words[9] == "the");

  insertion_sort(words, 8, 12, 0);

  BOOST_TEST(words[8] == "are");
  BOOST_TEST(words[9] == "seashells");
  BOOST_TEST(words[10] == "shells");
  BOOST_TEST(words[11] == "surely");
  BOOST_TEST(words[12] == "the");
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MostSignificantDigitFirstSortSorts)
{
  Array<string> words {create_different_length_strings()};

  most_significant_digit_first_sort(words);

  BOOST_TEST(words[0] == "are");
  BOOST_TEST(words[1] == "by");
  BOOST_TEST(words[2] == "seashells");
  BOOST_TEST(words[3] == "seashells");
  BOOST_TEST(words[4] == "seashore");
  BOOST_TEST(words[5] == "sells");
  BOOST_TEST(words[6] == "sells");
  BOOST_TEST(words[7] == "she");
  BOOST_TEST(words[8] == "she");
  BOOST_TEST(words[9] == "shells");
  BOOST_TEST(words[10] == "surely");
  BOOST_TEST(words[11] == "the");
  BOOST_TEST(words[12] == "the");
}

BOOST_AUTO_TEST_SUITE_END() // Strings
BOOST_AUTO_TEST_SUITE_END() // Sorting
BOOST_AUTO_TEST_SUITE_END() // Algorithms