#include "Algorithms/BinarySearch.h"

#include <boost/test/unit_test.hpp>
#include <vector>

using Algorithms::Search::Details::binary_search_first_occurrence_recursive;
using Algorithms::Search::Details::binary_search_last_occurrence_recursive;
using Algorithms::Search::binary_search_iterative;
using Algorithms::Search::binary_search_recursive;
using Algorithms::Search::find_number_of_occurrences_in_sorted_array;
using std::vector;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(BinarySearch_tests)

// cf. https://codereview.stackexchange.com/questions/234897/binary-search-given-sorted-array-unit-test
const vector<int> test_case_1 {
  1, 2, 4, 7, 8, 12, 15, 19, 24, 50, 69, 80, 100};

const vector<int> test_case_2 {1, 2, 6, 10};
const vector<int> test_case_3 {1, 2, 3, 3, 3, 3, 3, 3, 3};

// cf. https://leetcode.com/discuss/interview-question/algorithms/124724/facebook-onsite-count-occurrences-of-a-number-in-a-sorted-array
const vector<int> repeats_case_1 {4, 4, 8, 8, 8, 15, 16, 23, 23, 42};
const vector<int> repeats_case_2 {3, 5, 5, 5, 5, 7, 8, 8};

// cf. https://www.geeksforgeeks.org/count-number-of-occurrences-or-frequency-in-a-sorted-array/
const vector<int> repeats_case_3 {1, 1, 2, 2, 2, 2, 3};

BOOST_AUTO_TEST_SUITE(BinarySearchIterative_tests)

//------------------------------------------------------------------------------
/// \details Be careful of a huge error case that gets caught by a unit test
/// testing a value smaller than the left bound. The exit condition requires
/// that l > r. But what happens if l = 0 and the the type is unsigned
/// std::size_t? Underflow!
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ReturnNullOptOnLeftBound)
{
  BOOST_TEST(
    !static_cast<bool>(
      binary_search_iterative(test_case_1, 0, test_case_1.size() - 1)));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ReturnNullOptOnRightBound)
{
  BOOST_TEST(
    !static_cast<bool>(
      binary_search_iterative(test_case_1, 101, test_case_1.size() - 1)));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FindsElementPlacedOnLeftBound)
{
  const auto result = binary_search_iterative(
    test_case_1,
    1,
    test_case_1.size() - 1);

  BOOST_TEST(static_cast<bool>(result));

  BOOST_TEST(result.value() == 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FindsElementPlacedOnRightBound)
{
  const auto result = binary_search_iterative(
    test_case_1,
    100,
    test_case_1.size() - 1);

  BOOST_TEST(static_cast<bool>(result));

  BOOST_TEST(result.value() == 12);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FindsElementPlacedInMiddle)
{
  const auto result = binary_search_iterative(
    test_case_1,
    19,
    test_case_1.size() - 1);

  BOOST_TEST(static_cast<bool>(result));

  BOOST_TEST(result.value() == 7);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FindsElementPlacedLowerThanMiddle)
{
  const auto result = binary_search_iterative(
    test_case_1,
    12,
    test_case_1.size() - 1);

  BOOST_TEST(static_cast<bool>(result));

  BOOST_TEST(result.value() == 5);  
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FindsElementPlacedGreaterThanMiddle)
{
  const auto result = binary_search_iterative(
    test_case_1,
    69,
    test_case_1.size() - 1);

  BOOST_TEST(static_cast<bool>(result));

  BOOST_TEST(result.value() == 10);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ElementFromMiddleCannotBeFound)
{
  const auto result = binary_search_iterative(
    test_case_2,
    5,
    test_case_2.size() - 1);

  BOOST_TEST(!static_cast<bool>(result));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DoesNotDistinguishBetweenRepeatedValues)
{
  const auto result = binary_search_iterative(
    test_case_3,
    3,
    test_case_3.size() - 1);

  BOOST_TEST(static_cast<bool>(result));

  BOOST_TEST(result.value() == 4);
}

BOOST_AUTO_TEST_SUITE_END() // BinarySearchIterative_tests

BOOST_AUTO_TEST_SUITE(BinarySearchFirstOccurrenceRecursive_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ReturnNullOptOnLeftAndRightBound)
{
  BOOST_TEST(
    !static_cast<bool>(
      binary_search_first_occurrence_recursive(
        test_case_1,
        0,
        test_case_1.size() - 1)));

  BOOST_TEST(
    !static_cast<bool>(
      binary_search_first_occurrence_recursive(
        test_case_1,
        101,
        test_case_1.size() - 1)));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FindsUniqueValues)
{
  {
    const auto result = binary_search_first_occurrence_recursive(
      test_case_1,
      1,
      test_case_1.size() - 1);

    BOOST_TEST(static_cast<bool>(result));
    BOOST_TEST(result.value() == 0);
  }
  {
    const auto result = binary_search_first_occurrence_recursive(
      test_case_1,
      100,
      test_case_1.size() - 1);

    BOOST_TEST(static_cast<bool>(result));
    BOOST_TEST(result.value() == 12);
  }
  {
    const auto result = binary_search_first_occurrence_recursive(
      test_case_1,
      19,
      test_case_1.size() - 1);

    BOOST_TEST(static_cast<bool>(result));
    BOOST_TEST(result.value() == 7);
  }
  {
    const auto result = binary_search_first_occurrence_recursive(
      test_case_1,
      12,
      test_case_1.size() - 1);

    BOOST_TEST(static_cast<bool>(result));
    BOOST_TEST(result.value() == 5);
  }
  {
    const auto result = binary_search_first_occurrence_recursive(
      test_case_1,
      69,
      test_case_1.size() - 1);

    BOOST_TEST(static_cast<bool>(result));
    BOOST_TEST(result.value() == 10);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FindsFirstOccurrence)
{
  {
    const auto result = binary_search_first_occurrence_recursive(
      repeats_case_1,
      8,
      repeats_case_1.size() - 1);

    BOOST_TEST(static_cast<bool>(result));
    BOOST_TEST(result.value() == 2);
  }
  {
    const auto result = binary_search_first_occurrence_recursive(
      repeats_case_2,
      5,
      repeats_case_2.size() - 1);

    BOOST_TEST(static_cast<bool>(result));
    BOOST_TEST(result.value() == 1);
  }
}

BOOST_AUTO_TEST_SUITE_END() // BinarySearchFirstOccurrenceRecursive_tests

BOOST_AUTO_TEST_SUITE(FindNumberOfOccurrencesInSortedArray_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(WorksOnRecursiveSearch)
{
  {
    const auto result = find_number_of_occurrences_in_sorted_array(
      binary_search_first_occurrence_recursive<int, std::vector<int>>,
      binary_search_last_occurrence_recursive<int, std::vector<int>>,
      repeats_case_1,
      8,
      repeats_case_1.size() - 1);

    BOOST_TEST(result == 3);
  }
  {
    const auto result = find_number_of_occurrences_in_sorted_array(
      binary_search_first_occurrence_recursive<int, std::vector<int>>,
      binary_search_last_occurrence_recursive<int, std::vector<int>>,
      repeats_case_2,
      6,
      repeats_case_2.size() - 1);

    BOOST_TEST(result == 0);
  }
  {
    const auto result = find_number_of_occurrences_in_sorted_array(
      binary_search_first_occurrence_recursive<int, std::vector<int>>,
      binary_search_last_occurrence_recursive<int, std::vector<int>>,
      repeats_case_2,
      5,
      repeats_case_2.size() - 1);

    BOOST_TEST(result == 4);
  }
  {
    const auto result = find_number_of_occurrences_in_sorted_array(
      binary_search_first_occurrence_recursive<int, std::vector<int>>,
      binary_search_last_occurrence_recursive<int, std::vector<int>>,
      repeats_case_3,
      1,
      repeats_case_3.size() - 1);

    BOOST_TEST(result == 2);
  }
  {
    const auto result = find_number_of_occurrences_in_sorted_array(
      binary_search_first_occurrence_recursive<int, std::vector<int>>,
      binary_search_last_occurrence_recursive<int, std::vector<int>>,
      repeats_case_3,
      2,
      repeats_case_3.size() - 1);

    BOOST_TEST(result == 4);
  }
  {
    const auto result = find_number_of_occurrences_in_sorted_array(
      binary_search_first_occurrence_recursive<int, std::vector<int>>,
      binary_search_last_occurrence_recursive<int, std::vector<int>>,
      repeats_case_3,
      3,
      repeats_case_3.size() - 1);

    BOOST_TEST(result == 1);
  }
  {
    const auto result = find_number_of_occurrences_in_sorted_array(
      binary_search_first_occurrence_recursive<int, std::vector<int>>,
      binary_search_last_occurrence_recursive<int, std::vector<int>>,
      repeats_case_3,
      4,
      repeats_case_3.size() - 1);

    BOOST_TEST(result == 0);
  }
}

BOOST_AUTO_TEST_SUITE_END() // FindNumberOfOccurrencesInSortedArray_tests

BOOST_AUTO_TEST_SUITE_END() // BinarySearch_tests
BOOST_AUTO_TEST_SUITE_END() // Algorithms