#include "Algorithms/PreEasyExercises.h"

#include <boost/test/unit_test.hpp>
#include <vector>

using Algorithms::PreEasyExercises::ArrayIndexing;
using std::vector;

BOOST_AUTO_TEST_SUITE(Algorithms)

// https://blog.faangshui.com/p/before-leetcode

BOOST_AUTO_TEST_SUITE(PreEasyExercises)

// 1. Array Indexing
// Understanding how to navigate arrays is essential. Here are ten exercises,
// sorted in increasing difficulty, that build upon each other:

BOOST_AUTO_TEST_SUITE(ArrayIndexing_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IterateOverArrayIteratesOverAnArray)
{
  vector<int> array {1, 2, 3, 4, 5};
  BOOST_TEST(ArrayIndexing::iterate_over_array(array) == array);

  // ArrayIndexing::iterate_over_array(array, true);
}

//------------------------------------------------------------------------------
/// Iterate Over an Array in Reverse
/// Modify the previous function to print the elements in reverse order, from
/// the last to the first.
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IterateOverArrayInReverseIteratesOverAnArrayInReverse)
{
  vector<int> array {1, 2, 3, 4, 5};
  vector<int> expected_result {5, 4, 3, 2, 1};

  BOOST_TEST(
    ArrayIndexing::iterate_over_array_in_reverse(array) == expected_result);

  // ArrayIndexing::iterate_over_array_in_reverse(array, true);
}

BOOST_AUTO_TEST_SUITE_END() // ArrayIndexing_tests

BOOST_AUTO_TEST_SUITE_END() // PreEasyExercises
BOOST_AUTO_TEST_SUITE_END() // Algorithms