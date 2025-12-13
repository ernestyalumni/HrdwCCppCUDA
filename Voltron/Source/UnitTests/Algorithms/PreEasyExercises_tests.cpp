#include "Algorithms/PreEasyExercises.h"

#include <boost/test/unit_test.hpp>
#include <tuple>
#include <vector>

using Algorithms::PreEasyExercises::AccumulatorVariables;
using Algorithms::PreEasyExercises::ArrayIndexing;
using std::get;
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

//------------------------------------------------------------------------------
/// 8. Traverse the Perimeter of a Matrix
/// Print the elements along the outer edge (perimeter) of a 2D array.
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TraversePerimeterTraversesPerimeter)
{
  vector<int> expected_result {1, 2, 3, 6, 9, 8, 7, 4};
  BOOST_TEST(
    ArrayIndexing::traverse_perimeter(matrix) == expected_result);

  expected_result = {11, 12, 13, 14, 24, 34, 44, 43, 42, 41, 31, 21};
  BOOST_TEST(
    ArrayIndexing::traverse_perimeter(matrix_2) == expected_result);
}

//------------------------------------------------------------------------------
/// 9. Traverse Elements in Spiral Order
/// Print elements of a 2D array in spiral order, starting from the top-left
/// corner and moving inward.
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TraverseSpiralClockwiseTraversesClockwise)
{
  vector<int> expected_result {1, 2, 3, 6, 9, 8, 7, 4, 5};
  BOOST_TEST(
    ArrayIndexing::traverse_spiral_clockwise(matrix) == expected_result);

  expected_result = {
    11, 12, 13, 14, 24, 34, 44, 43, 42, 41, 31, 21, 22, 23, 33, 32};
  BOOST_TEST(
    ArrayIndexing::traverse_spiral_clockwise(matrix_2) == expected_result);
}

//------------------------------------------------------------------------------
/// 10. Traverse the Lower Triangle of a Matrix
/// Print the elements below and including the main diagonal of a square matrix.
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TraverseLowerTriangleTraverses)
{
  vector<int> expected_result {1, 4, 5, 7, 8, 9};
  BOOST_TEST(
    ArrayIndexing::traverse_lower_triangle(matrix) == expected_result);

  expected_result = {11, 21, 22, 31, 32, 33, 41, 42, 43, 44};
  BOOST_TEST(
    ArrayIndexing::traverse_lower_triangle(matrix_2) == expected_result);
}

//------------------------------------------------------------------------------
/// https://blog.faangshui.com/p/before-leetcode
/// 2. Accumulator Variables
/// Learn how to keep track of values during iteration.
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
/// 1. Calculate the Sum of an Array
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CalculateSumCalculatesSum)
{
  BOOST_TEST(AccumulatorVariables::calculate_sum(array) == 15);
  BOOST_TEST(AccumulatorVariables::calculate_sum(array_2) == 21);
  BOOST_TEST(AccumulatorVariables::calculate_sum(all_same) == 20);
}

//------------------------------------------------------------------------------
/// 2. Find the Minimum and Maximum Elements
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FindMinMaxFindsMinMax)
{
  BOOST_TEST(get<0>(AccumulatorVariables::find_min_max(array)) == 1);
  BOOST_TEST(get<1>(AccumulatorVariables::find_min_max(array)) == 5);

  BOOST_TEST(get<0>(AccumulatorVariables::find_min_max(array_2)) == 1);
  BOOST_TEST(get<1>(AccumulatorVariables::find_min_max(array_2)) == 9);

  BOOST_TEST(get<0>(AccumulatorVariables::find_min_max(all_same)) == 5);
  BOOST_TEST(get<1>(AccumulatorVariables::find_min_max(all_same)) == 5);
}
//------------------------------------------------------------------------------
/// 3. Find the Indices of the Min and Max Elements
//------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(FindMinMaxIndicesFindsIndices)
{
  BOOST_TEST(get<0>(AccumulatorVariables::find_min_max_indices(array)) == 0);
  BOOST_TEST(get<1>(AccumulatorVariables::find_min_max_indices(array)) == 4);

  BOOST_TEST(get<0>(AccumulatorVariables::find_min_max_indices(array_2)) == 3);
  BOOST_TEST(get<1>(AccumulatorVariables::find_min_max_indices(array_2)) == 2);

  BOOST_TEST(get<0>(AccumulatorVariables::find_min_max_indices(all_same)) == 0);
  BOOST_TEST(get<1>(AccumulatorVariables::find_min_max_indices(all_same)) == 0);
}

//------------------------------------------------------------------------------
/// 4. Find the Two Smallest/Largest Elements Without Sorting
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FindTwoMinMaxIndicesFindsTwoMinMaxIndices)
{
  BOOST_TEST(
    get<0>(AccumulatorVariables::find_two_min_max_indices(array)) == 0);
  BOOST_TEST(
    get<1>(AccumulatorVariables::find_two_min_max_indices(array)) == 1);
  BOOST_TEST(
    get<2>(AccumulatorVariables::find_two_min_max_indices(array)) == 4);
  BOOST_TEST(
    get<3>(AccumulatorVariables::find_two_min_max_indices(array)) == 3);

  BOOST_TEST(
    get<0>(AccumulatorVariables::find_two_min_max_indices(array_2)) == 3);
  BOOST_TEST(
    get<1>(AccumulatorVariables::find_two_min_max_indices(array_2)) == 0);
  BOOST_TEST(
    get<2>(AccumulatorVariables::find_two_min_max_indices(array_2)) == 2);
  BOOST_TEST(
    get<3>(AccumulatorVariables::find_two_min_max_indices(array_2)) == 1);

  BOOST_TEST(
    get<0>(AccumulatorVariables::find_two_min_max_indices(all_same)) == 0);
  BOOST_TEST(
    get<1>(AccumulatorVariables::find_two_min_max_indices(all_same)) == 1);
  BOOST_TEST(
    get<2>(AccumulatorVariables::find_two_min_max_indices(all_same)) == 0);
  BOOST_TEST(
    get<3>(AccumulatorVariables::find_two_min_max_indices(all_same)) == 1);
}

//------------------------------------------------------------------------------
/// 5. Count Occurrences of a Specific Element
//------------------------------------------------------------------------------


BOOST_AUTO_TEST_SUITE_END() // ArrayIndexing_tests

BOOST_AUTO_TEST_SUITE_END() // PreEasyExercises
BOOST_AUTO_TEST_SUITE_END() // Algorithms