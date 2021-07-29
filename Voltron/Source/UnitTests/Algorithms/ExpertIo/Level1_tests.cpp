#include "Algorithms/ExpertIo/Level1.h"

#include <boost/test/unit_test.hpp>
#include <string>
#include <vector>

using Algorithms::ExpertIo::EasyNonConstructibleChange::
  non_constructible_change_sort;
using Algorithms::ExpertIo::NthFibonacci::get_nth_fibonacci_recursive;
using Algorithms::ExpertIo::is_valid_subsequence;
using Algorithms::ExpertIo::sorted_squared_array_algorithmic;
using Algorithms::ExpertIo::sorted_squared_array_two_indices;
using Algorithms::ExpertIo::sorted_squared_array_with_selection_sort;
using Algorithms::ExpertIo::tournament_winner;
using Algorithms::ExpertIo::two_number_sum_brute;
using Algorithms::ExpertIo::two_number_sum_with_map;

using std::string;
using std::vector;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(ExpertIo)

BOOST_AUTO_TEST_SUITE(TwoNumberSum)

const vector<int> example_array {3, 5, -4, 8, 11, 1, -1, 6};
const int example_target_sum {10};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ExampleFindingAnyTwoArraySumBruteForce)
{
  const vector<int> brute_solution {
    two_number_sum_brute(example_array, example_target_sum)};

  BOOST_TEST(brute_solution[0] == 11);
  BOOST_TEST(brute_solution[1] == -1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ExampleFindingAnyTwoArraySumWithMap)
{
  const vector<int> map_solution {
    two_number_sum_with_map(example_array, example_target_sum)};

  BOOST_TEST(map_solution[0] == 11);
  BOOST_TEST(map_solution[1] == -1);
}

BOOST_AUTO_TEST_SUITE_END() // TwoNumberSum

BOOST_AUTO_TEST_SUITE(ValidateSubsequence)

const vector<int> sample_array {5, 1, 22, 25, 6, -1, 8, 10};
const vector<int> sample_sequence {1, 6, -1, 10};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ExampleWithTwoPointers)
{
  BOOST_TEST(is_valid_subsequence(sample_array, sample_sequence));
}

BOOST_AUTO_TEST_SUITE_END() // ValidateSubsequence

BOOST_AUTO_TEST_SUITE(SortedSquaredArray)

const vector<int> sample_array {1, 2, 3, 4, 5, 6, 8, 9};

const vector<int> test_case_array_8 {-2, -1};

const vector<int> test_case_array_9 {-5, -4, -3, -2, -1};

const vector<int> test_case_array_11 {-10, -5, 0, 5, 10};

const vector<int> test_case_array_12 {-7, -3, 1, 9, 22, 30};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SortedSquaredArrayWithAlgorithmImplementation)
{
  const auto x = sorted_squared_array_algorithmic(sample_array);

  auto iter_x = x.begin();

  for (const auto& a : sample_array)
  {
    BOOST_TEST((a * a == *iter_x));

    ++iter_x;
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SquareOfNegativeValuesGetsSorted)
{
  {
    const auto x = sorted_squared_array_algorithmic(test_case_array_8);

    BOOST_TEST(x.at(0) == 1);
    BOOST_TEST(x.at(1) == 4);
  }

}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SquareOfNegativeValuesGetsSortedWithSelectionSort)
{
  {
    const auto x = sorted_squared_array_with_selection_sort(test_case_array_9);

    BOOST_TEST(x.at(0) == 1);
    BOOST_TEST(x.at(1) == 4);
    BOOST_TEST(x.at(2) == 9);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SquareOfPositiveValuesGetsSortedWithTwoIndices)
{
  {
    const auto x = sorted_squared_array_two_indices(sample_array);

    BOOST_TEST(x.at(0) == 1);
    BOOST_TEST(x.at(1) == 4);
    BOOST_TEST(x.at(2) == 9);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SquareOfNegativeValuesGetsSortedWithTwoIndices)
{
  {
    const auto x = sorted_squared_array_two_indices(test_case_array_8);

    BOOST_TEST(x.at(0) == 1);
    BOOST_TEST(x.at(1) == 4);
  }
  {
    const auto x = sorted_squared_array_two_indices(test_case_array_9);

    BOOST_TEST(x.at(0) == 1);
    BOOST_TEST(x.at(1) == 4);
    BOOST_TEST(x.at(2) == 9);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SquareOValuesGetsSortedWithTwoIndices)
{
  {
    const auto x = sorted_squared_array_two_indices(test_case_array_11);

    BOOST_TEST(x.at(0) == 0);
    BOOST_TEST(x.at(1) == 25);
    BOOST_TEST(x.at(2) == 25);
  }
  {
    const auto x = sorted_squared_array_two_indices(test_case_array_12);

    BOOST_TEST(x.at(0) == 1);
    BOOST_TEST(x.at(1) == 9);
    BOOST_TEST(x.at(2) == 49);
  }
}

BOOST_AUTO_TEST_SUITE_END() // SortedSquaredArray

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TournamentWinnerWithUnorderedMapWorks)
{
  const vector<vector<string>> competitions {
    {"HTML", "C#"},
    {"C#", "Python"},
    {"Python", "HTML"}};

  const vector<int> competition_results {0, 0, 1};

  const string winner {tournament_winner(competitions, competition_results)};

  BOOST_TEST(winner == "Python");
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TournamentWinnerWithUnorderedMapWorksForOtherTestCases)
{
  // Test case 9
  {
    const vector<vector<string>> competitions {
      {"HTML", "Java"},
      {"Java", "Python"},
      {"Python", "HTML"},
      {"C#", "Python"},
      {"Java", "C#"},
      {"C#", "HTML"},
      {"SQL", "C#"},
      {"HTML", "SQL"},
      {"SQL", "Python"},
      {"SQL", "Java"}};

    const vector<int> competition_results {0, 0, 0, 0, 0, 0, 1, 0, 1, 1};

    const string winner {tournament_winner(competitions, competition_results)};

    BOOST_TEST(winner == "SQL");
  }

  // Test case 10
  {
    const vector<vector<string>> competitions {
      {"A", "B"}};

    const vector<int> competition_results {0};

    const string winner {tournament_winner(competitions, competition_results)};

    BOOST_TEST(winner == "B");
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(NonConstructibleChangeSortPassesSampleCases)
{
  {
    const vector<int> given_coins {1, 2, 5};
    const int expected {4};

    const int result {non_constructible_change_sort(given_coins)};

    BOOST_TEST(result == 6);
  }

}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(NonConstructibleChangePassesSampleCases)
{
  {
    const vector<int> given_coins {1, 2, 5};
    const int expected {4};


  }

}

BOOST_AUTO_TEST_SUITE(NthFibonacci)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FibonacciRecursiveComputesCorrectly)
{
  BOOST_TEST(get_nth_fibonacci_recursive(1) == 0);
  BOOST_TEST(get_nth_fibonacci_recursive(2) == 1);
  BOOST_TEST(get_nth_fibonacci_recursive(16) == 610);
}

BOOST_AUTO_TEST_SUITE_END() // NthFibonacci

BOOST_AUTO_TEST_SUITE_END() // ExpertIo
BOOST_AUTO_TEST_SUITE_END() // Algorithms