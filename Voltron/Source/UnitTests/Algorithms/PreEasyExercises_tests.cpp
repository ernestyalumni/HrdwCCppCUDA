#include "Algorithms/PreEasyExercises.h"

#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

using Algorithms::PreEasyExercises::AccumulatorVariables;
using Algorithms::PreEasyExercises::ArrayIndexing;
using Algorithms::PreEasyExercises::Recursion;
using std::get;
using std::unordered_set;
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
BOOST_AUTO_TEST_CASE(CountOccurrencesCountsOccurrences)
{
  BOOST_TEST(AccumulatorVariables::count_occurrences(array, 1) == 1);
  BOOST_TEST(AccumulatorVariables::count_occurrences(array, 2) == 1);
  BOOST_TEST(AccumulatorVariables::count_occurrences(array, 3) == 1);
  BOOST_TEST(AccumulatorVariables::count_occurrences(array, 0) == 0);
  BOOST_TEST(AccumulatorVariables::count_occurrences(array_2, 4) == 1);
  BOOST_TEST(AccumulatorVariables::count_occurrences(array_2, 7) == 1);
  BOOST_TEST(AccumulatorVariables::count_occurrences(array_2, 13) == 0);
  BOOST_TEST(AccumulatorVariables::count_occurrences(all_same, 5) == 4);
  BOOST_TEST(AccumulatorVariables::count_occurrences(all_same, 6) == 0);
}

//------------------------------------------------------------------------------
/// 6. Count Occurrences of All Elements
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CountOccurrencesOfAllCountsOccurrencesOfAll)
{
  auto result = AccumulatorVariables::count_occurrences_of_all(array);
  BOOST_TEST(result[1] == 1);
  BOOST_TEST(result[2] == 1);
  BOOST_TEST(result[3] == 1);
  BOOST_TEST(result[4] == 1);
  BOOST_TEST(result[5] == 1);
  BOOST_TEST(result[0] == 0);
  BOOST_TEST(result[6] == 0);
  BOOST_TEST(result[7] == 0);

  result = AccumulatorVariables::count_occurrences_of_all(array_2);
  BOOST_TEST(result[4] == 1);
  BOOST_TEST(result[7] == 1);
  BOOST_TEST(result[9] == 1);
  BOOST_TEST(result[1] == 1);
  BOOST_TEST(result[21] == 0);
  BOOST_TEST(result[22] == 0);
  BOOST_TEST(result[23] == 0);
  BOOST_TEST(result[24] == 0);

  result = AccumulatorVariables::count_occurrences_of_all(all_same);
  BOOST_TEST(result[5] == 4);
  BOOST_TEST(result[6] == 0);
}

//------------------------------------------------------------------------------
/// 7. Find the Two Most Frequent Elements
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FindTwoMostFrequentFindsTwoMostFrequent)
{
  BOOST_TEST(get<0>(AccumulatorVariables::find_two_most_frequent(array)) == 5);
  BOOST_TEST(get<1>(AccumulatorVariables::find_two_most_frequent(array)) == 4);

  BOOST_TEST(get<0>(AccumulatorVariables::find_two_most_frequent(array_2)) == 1);
  BOOST_TEST(get<1>(AccumulatorVariables::find_two_most_frequent(array_2)) == 9);

  BOOST_TEST(get<0>(AccumulatorVariables::find_two_most_frequent(all_same)) == 5);
  BOOST_TEST(get<1>(AccumulatorVariables::find_two_most_frequent(all_same)) == 5);
}

//------------------------------------------------------------------------------
/// 8. Compute Prefix Sums
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ComputePrefixSumsComputesPrefixSums)
{
  BOOST_TEST(
    (AccumulatorVariables::compute_prefix_sums(array) ==
      vector<int> {1, 3, 6, 10, 15}));
  BOOST_TEST(
    (AccumulatorVariables::compute_prefix_sums(array_2) ==
      vector<int> {4, 11, 20, 21}));
  BOOST_TEST(
    (AccumulatorVariables::compute_prefix_sums(all_same) ==
      vector<int> {5, 10, 15, 20}));
}

//------------------------------------------------------------------------------
/// 9. Find the Sum of Elements in a Given Range
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FindSumInRangeFindsSumInRange)
{
  BOOST_TEST(AccumulatorVariables::find_sum_in_range(array, 0, 4) == 15);
  BOOST_TEST(AccumulatorVariables::find_sum_in_range(array_2, 0, 3) == 21);
  BOOST_TEST(AccumulatorVariables::find_sum_in_range(all_same, 0, 3) == 20);

  BOOST_TEST(AccumulatorVariables::find_sum_in_range(array, 1, 3) == 9);
  BOOST_TEST(AccumulatorVariables::find_sum_in_range(array_2, 1, 2) == 16);
  BOOST_TEST(AccumulatorVariables::find_sum_in_range(all_same, 1, 2) == 10);
}

//------------------------------------------------------------------------------
/// 10. Efficient Range Sum Queries Using Prefix Sums
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(EfficientRangeSumQueriesFindsSumInRange)
{
  BOOST_TEST(
    AccumulatorVariables::efficient_range_sum_queries(array, 0, 4) == 15);
  BOOST_TEST(
    AccumulatorVariables::efficient_range_sum_queries(array_2, 0, 3) == 21);
  BOOST_TEST(
    AccumulatorVariables::efficient_range_sum_queries(all_same, 0, 3) == 20);

  BOOST_TEST(
    AccumulatorVariables::efficient_range_sum_queries(array, 1, 3) == 9);
  BOOST_TEST(
    AccumulatorVariables::efficient_range_sum_queries(array_2, 1, 2) == 16);
  BOOST_TEST(
    AccumulatorVariables::efficient_range_sum_queries(all_same, 1, 2) == 10);

  BOOST_TEST(
    AccumulatorVariables::efficient_range_sum_queries(array, 0, 0) == 1);
  BOOST_TEST(
    AccumulatorVariables::efficient_range_sum_queries(array_2, 0, 0) == 4);
  BOOST_TEST(
    AccumulatorVariables::efficient_range_sum_queries(all_same, 0, 0) == 5);

  BOOST_TEST(
    AccumulatorVariables::efficient_range_sum_queries(array, 4, 4) == 5);
  BOOST_TEST(
    AccumulatorVariables::efficient_range_sum_queries(array_2, 3, 3) == 1);
  BOOST_TEST(
    AccumulatorVariables::efficient_range_sum_queries(array_2, 3, 3) == 1);
  BOOST_TEST(
    AccumulatorVariables::efficient_range_sum_queries(all_same, 3, 3) == 5);
}

//------------------------------------------------------------------------------
/// 1. Calculate the nth Fibonacci Number
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CalculateNthFibonacciNumberFindsNthFibonacciNumber)
{
  BOOST_TEST(Recursion::calculate_fibonacci_number(0) == 1);
  BOOST_TEST(Recursion::calculate_fibonacci_number(1) == 1);
  BOOST_TEST(Recursion::calculate_fibonacci_number(2) == 1);
  BOOST_TEST(Recursion::calculate_fibonacci_number(3) == 2);
  BOOST_TEST(Recursion::calculate_fibonacci_number(13) == 233);
  BOOST_TEST(Recursion::calculate_fibonacci_number(14) == 377);
}

//------------------------------------------------------------------------------
/// 2. Sum of an Array Using Recursion
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SumRecursiveSums)
{
  BOOST_TEST(Recursion::sum_recursive(array) == 15);
  BOOST_TEST(Recursion::sum_recursive(array_2) == 21);
  BOOST_TEST(Recursion::sum_recursive(all_same) == 20);
}

//------------------------------------------------------------------------------
/// 3. Find the Minimum Element in an Array Using Recursion
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FindMinFindsMin)
{
  BOOST_TEST(Recursion::find_min(array) == 1);
  BOOST_TEST(Recursion::find_min(array_2) == 1);
  BOOST_TEST(Recursion::find_min(all_same) == 5);
}

//------------------------------------------------------------------------------
/// 4. Reverse a String Using Recursion
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ReverseStringReversesString)
{
  vector<int> expected_result {5, 4, 3, 2, 1};
  Recursion::reverse_string(array);
  BOOST_TEST((array == expected_result));

  std::string test_string {"hello"};
  Recursion::reverse_string(test_string);
  BOOST_TEST((test_string == "olleh"));

  test_string = "ab";
  Recursion::reverse_string(test_string);
  BOOST_TEST((test_string == "ba"));

  test_string = "a";
  Recursion::reverse_string(test_string);
  BOOST_TEST((test_string == "a"));

  test_string = "";
  Recursion::reverse_string(test_string);
  BOOST_TEST((test_string == ""));
}

//------------------------------------------------------------------------------
/// 5. Check if a String is a Palindrome Using Recursion
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IsPalindromeChecksIfStringIsPalindrome)
{
  BOOST_TEST(Recursion::is_palindrome(array) == false);
  BOOST_TEST(Recursion::is_palindrome(array_2) == false);
  BOOST_TEST(Recursion::is_palindrome(all_same) == true);

  std::string test_string {"hello"};
  BOOST_TEST(Recursion::is_palindrome(test_string) == false);

  test_string = "ab";
  BOOST_TEST(Recursion::is_palindrome(test_string) == false);

  test_string = "a";
  BOOST_TEST(Recursion::is_palindrome(test_string) == true);

  test_string = "";
  BOOST_TEST(Recursion::is_palindrome(test_string) == true);

  test_string = "aba";
  BOOST_TEST(Recursion::is_palindrome(test_string) == true);

  test_string = "abba";
  BOOST_TEST(Recursion::is_palindrome(test_string) == true);

  test_string = "abcdcba";
  BOOST_TEST(Recursion::is_palindrome(test_string) == true);
}

//------------------------------------------------------------------------------
/// 6. Generate All Permutations of a String
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(GenerateAllPermutationsGeneratesAllPermutations)
{
  {
    std::string test_string {""};
    const auto permutations = Recursion::generate_all_permutations(test_string);

    BOOST_TEST(permutations.size() == 0);
  }
  {
    std::string test_string {"a"};
    const auto permutations = Recursion::generate_all_permutations(test_string);

    BOOST_TEST(permutations.size() == 1);
    BOOST_TEST(permutations[0] == "a");
  }
  {
    std::string test_string {"ab"};
    const auto permutations = Recursion::generate_all_permutations(test_string);

    BOOST_TEST(permutations.size() == 2);
    std::unordered_set<std::string> expected_permutations {"ab", "ba"};
    for (const auto& permutation : permutations)
    {
      BOOST_TEST(expected_permutations.contains(permutation));
    }
  }
  {
    std::string test_string {"abc"};
    const auto permutations = Recursion::generate_all_permutations(test_string);

    BOOST_TEST(permutations.size() == 6);
    std::unordered_set<std::string> expected_permutations {
      "abc", "acb", "bac", "bca", "cab", "cba"};
    for (const auto& permutation : permutations)
    {
      BOOST_TEST(expected_permutations.contains(permutation));
    }
  }
  {
    std::string test_string {"aba"};
    const auto permutations = Recursion::generate_all_permutations(test_string);

    BOOST_TEST(permutations.size() == 6);
    std::unordered_set<std::string> expected_permutations {"aba", "aab", "baa"};
    for (const auto& permutation : permutations)
    {
      BOOST_TEST(expected_permutations.contains(permutation));
    }
  }
}

class GenerateAllSubsetsTestCases
{
  public:
    static unordered_set<int> get_test_set_1()
    {
      return {1, 2, 3};
    }

    static vector<unordered_set<int>> get_expected_subsets_1()
    {
      return {{}, {1}, {2}, {3}, {1, 2}, {1, 3}, {2, 3}, {1, 2, 3}};
    }

    static bool test_power_set_against_expected(
      const auto& power_set,
      const auto& expected_subsets)
    {
      bool all_matches_found {true};
      for (const auto& subset : power_set)
      {
        bool found_match {false};
        for (const auto& expected : expected_subsets)
        {
          if (subset.size() == expected.size())
          {
            bool matches {true};
            for (const auto& elem : subset)
            {
              if (expected.find(elem) == expected.end())
              {
                matches = false;
                break;
              }
            }
            if (matches)
            {
              found_match = true;
              break;
            }
          }
        }
        if (not found_match)
        {
          all_matches_found = false;
          break;
        }
      }
      return all_matches_found;
    }
};

//------------------------------------------------------------------------------
/// 7. Generate All Subsets of a Set
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(GenerateAllSubsetsWithBitfieldGeneratesPowerSet)
{
  {
    const auto test_set = GenerateAllSubsetsTestCases::get_test_set_1();
    const auto power_set = Recursion::generate_all_subsets_with_bitfield(
      test_set);

    // 2^3 = 8 subsets
    BOOST_TEST(power_set.size() == 8);

    // Check that each generated subset matches an expected one
    BOOST_TEST(GenerateAllSubsetsTestCases::test_power_set_against_expected(
      power_set,
      GenerateAllSubsetsTestCases::get_expected_subsets_1()));

    // Also check that we have the right number of each size
    std::unordered_map<std::size_t, std::size_t> size_counts {};
    for (const auto& subset : power_set)
    {
      size_counts[subset.size()]++;
    }
    BOOST_TEST(size_counts[0] == 1);  // 1 empty set
    BOOST_TEST(size_counts[1] == 3);  // 3 single-element sets
    BOOST_TEST(size_counts[2] == 3);  // 3 two-element sets
    BOOST_TEST(size_counts[3] == 1);  // 1 three-element set
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(GenerateAllSubsetsGeneratesPowerSet)
{
  {
    const auto test_set = GenerateAllSubsetsTestCases::get_test_set_1();
    const auto power_set = Recursion::generate_all_subsets(test_set);

    // 2^3 = 8 subsets
    BOOST_TEST(power_set.size() == 8);

    BOOST_TEST(GenerateAllSubsetsTestCases::test_power_set_against_expected(
      power_set,
      GenerateAllSubsetsTestCases::get_expected_subsets_1()));

    // Also check that we have the right number of each size
    std::unordered_map<std::size_t, std::size_t> size_counts {};
    for (const auto& subset : power_set)
    {
      size_counts[subset.size()]++;
    }
    BOOST_TEST(size_counts[0] == 1);  // 1 empty set
    BOOST_TEST(size_counts[1] == 3);  // 3 single-element sets
    BOOST_TEST(size_counts[2] == 3);  // 3 two-element sets
    BOOST_TEST(size_counts[3] == 1);  // 1 three-element set
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(GenerateAllSubsetsIterativelyGeneratesPowerSet)
{
  {
    const auto test_set = GenerateAllSubsetsTestCases::get_test_set_1();
    const auto power_set = Recursion::generate_all_subsets(test_set);

    // 2^3 = 8 subsets
    BOOST_TEST(power_set.size() == 8);

    BOOST_TEST(GenerateAllSubsetsTestCases::test_power_set_against_expected(
      power_set,
      GenerateAllSubsetsTestCases::get_expected_subsets_1()));

    // Also check that we have the right number of each size
    std::unordered_map<std::size_t, std::size_t> size_counts {};
    for (const auto& subset : power_set)
    {
      size_counts[subset.size()]++;
    }
    BOOST_TEST(size_counts[0] == 1);  // 1 empty set
    BOOST_TEST(size_counts[1] == 3);  // 3 single-element sets
    BOOST_TEST(size_counts[2] == 3);  // 3 two-element sets
    BOOST_TEST(size_counts[3] == 1);  // 1 three-element set
  }
}

//------------------------------------------------------------------------------
/// 8. Compute the Sum of Digits of a Number
//------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(ComputeSumOfDigitsIterativelyComputesSumOfDigits)
{
  {
    const int n {12345};
    const unsigned int result {
      Recursion::compute_sum_of_digits_iteratively(n)};
    const unsigned int output {15};
    BOOST_TEST(result == output);
  }
  {
    const int n {10000};
    const unsigned int result {
      Recursion::compute_sum_of_digits_iteratively(n)};
    const unsigned int output {1};
    BOOST_TEST(result == output);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ComputeSumOfDigitsComputesSumOfDigitsRecursively)
{
  {
    const int n {12345};
    const unsigned int result {
      Recursion::compute_sum_of_digits(n)};
    const unsigned int output {15};
    BOOST_TEST(result == output);
  }
  {
    const int n {10000};
    const unsigned int result {
      Recursion::compute_sum_of_digits(n)};
    const unsigned int output {1};
    BOOST_TEST(result == output);
  }
}

//------------------------------------------------------------------------------
/// https://blog.faangshui.com/i/149072585/recursion
/// 9. Compute the Power of a Number
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ComputePowerOfNumberComputesPowerOfNumberRecursively)
{
  {
    const int x {2};
    const int n {3};
    const int result {
      Recursion::compute_power_of_number(x, n)};
    const int output {8};
    BOOST_TEST(result == output);
  }
  {
    const int x {3};
    const unsigned int n {4};
    const int result {
      Recursion::compute_power_of_number(x, n)};
    const int output {81};
    BOOST_TEST(result == output);
  }
}

//------------------------------------------------------------------------------
/// https://blog.faangshui.com/i/149072585/recursion
/// 10. Count the Number of Occurrences of a Character in a String Using
/// Recursion
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CountNumberOfOccurrencesOfACharacterCounts)
{
  {
    const std::string s {"hello"};
    const char c {'l'};
    const uint64_t result {
      Recursion::count_number_of_occurrences_of_a_character(s, c)};

    const uint64_t output {2};
    BOOST_TEST(result == output);

    const char c2 {'h'};
    const uint64_t result2 {
      Recursion::count_number_of_occurrences_of_a_character(s, c2)};

    const uint64_t output2 {1};
    BOOST_TEST(result2 == output2);
  }
  {
    const std::string s {"wrappworld"};
    const char c {'w'};
    const uint64_t result {
      Recursion::count_number_of_occurrences_of_a_character(s, c)};

    const uint64_t output {2};
    BOOST_TEST(result == output);

    const char c2 {'o'};
    const uint64_t result2 {
      Recursion::count_number_of_occurrences_of_a_character(s, c2)};

    const uint64_t output2 {1};
    BOOST_TEST(result2 == output2);
  }
}

BOOST_AUTO_TEST_SUITE_END() // ArrayIndexing_tests

BOOST_AUTO_TEST_SUITE_END() // PreEasyExercises
BOOST_AUTO_TEST_SUITE_END() // Algorithms