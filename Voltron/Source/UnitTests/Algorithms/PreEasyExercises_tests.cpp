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

vector<int> array {1, 2, 3, 4, 5};
vector<int> array_2 {4, 7, 9, 1};
vector<int> all_same {5, 5, 5, 5};

vector<vector<int>> matrix {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
vector<vector<int>> matrix_2 {
  {11, 12, 13, 14},
  {21, 22, 23, 24},
  {31, 32, 33, 34},
  {41, 42, 43, 44}
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IterateOverArrayIteratesOverAnArray)
{
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
  vector<int> expected_result {5, 4, 3, 2, 1};

  BOOST_TEST(
    ArrayIndexing::iterate_over_array_in_reverse(array) == expected_result);

  // ArrayIndexing::iterate_over_array_in_reverse(array, true);
}

//------------------------------------------------------------------------------
/// 3. Fetch Every Second Element
///
/// Write a function that accesses every other element in the array, starting
/// from the first element.
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FetchSecondElementFetchesSecondElement)
{
  vector<int> expected_result {1, 3, 5};
  BOOST_TEST(ArrayIndexing::fetch_second_element(array) == expected_result);
}

//------------------------------------------------------------------------------
/// 4. Find the Index of a Target Element
/// 
/// Write a function that searches for a specific element in an array and
/// returns its index. If the element is not found, return -1.
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FindIndexOfTargetElementFindsIndex)
{
  {
    BOOST_TEST(ArrayIndexing::find_index_of_target_element(array_2, 1) == 3);
    BOOST_TEST(ArrayIndexing::find_index_of_target_element(array_2, 9) == 2);
    BOOST_TEST(ArrayIndexing::find_index_of_target_element(array_2, 5) == -1);
  }

  vector<int> empty_array {};
  BOOST_TEST(ArrayIndexing::find_index_of_target_element(empty_array, 1) == -1);

  vector<int> single_element_array {1};
  BOOST_TEST(ArrayIndexing::find_index_of_target_element(single_element_array, 1) == 0);
  BOOST_TEST(ArrayIndexing::find_index_of_target_element(single_element_array, 2) == -1);

  vector<int> duplicates {1, 2, 2, 3, 2, 4};
  BOOST_TEST(ArrayIndexing::find_index_of_target_element(duplicates, 2) == 1);

  BOOST_TEST(ArrayIndexing::find_index_of_target_element(all_same, 5) == 0);
}

//------------------------------------------------------------------------------
/// 5. Find the First Prime Number in an Array
///
/// Iterate over an array and find the first prime number. Stop the iteration
/// once you find it.
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FindFirstPrimeNumberFindsFirstPrimeNumber)
{
  BOOST_TEST(ArrayIndexing::find_first_prime_number(array) == 2);
  BOOST_TEST(ArrayIndexing::find_first_prime_number(array_2) == 7);

  BOOST_TEST(ArrayIndexing::find_first_prime_number(all_same) == 5);
}

//------------------------------------------------------------------------------
/// 6. Traverse a Two-Dimensional Array
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TraverseTwoDimensionalArrayTraverses)
{
  vector<int> expected_result {1, 2, 3, 4, 5, 6, 7, 8, 9};
  BOOST_TEST(
    ArrayIndexing::traverse_two_dimensional_array(matrix) == expected_result);

  expected_result = {
    11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34, 41, 42, 43, 44};
  BOOST_TEST(
    ArrayIndexing::traverse_two_dimensional_array(matrix_2) == expected_result);
}

//------------------------------------------------------------------------------
/// 7. Traverse the Main Diagonal of a Matrix
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TraverseMainDiagonalTraverses)
{
  vector<int> expected_result {1, 5, 9};

  BOOST_TEST(
    ArrayIndexing::traverse_main_diagonal(matrix) == expected_result);

  expected_result = {11, 22, 33, 44};
  BOOST_TEST(
    ArrayIndexing::traverse_main_diagonal(matrix_2) == expected_result);
}

BOOST_AUTO_TEST_SUITE_END() // ArrayIndexing_tests

BOOST_AUTO_TEST_SUITE_END() // PreEasyExercises
BOOST_AUTO_TEST_SUITE_END() // Algorithms