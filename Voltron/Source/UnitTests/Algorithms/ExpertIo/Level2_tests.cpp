#include "Algorithms/ExpertIo/Level2.h"

#include <boost/test/unit_test.hpp>
#include <vector>

using Algorithms::ExpertIo::MergeOverlappingIntervals::
  insertion_sort_intervals;
using Algorithms::ExpertIo::MergeOverlappingIntervals::
  merge_overlapping_intervals;

using std::vector;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(ExpertIo)

BOOST_AUTO_TEST_SUITE(MergeOverlappingIntervals)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(InsertionSortIntervalSorts)
{
  {
    vector<vector<int>> input {{1, 2}, {3, 8}, {9, 10}};
    insertion_sort_intervals(input);
    BOOST_TEST(input[0][0] == 1);
    BOOST_TEST(input[1][0] == 3);
    BOOST_TEST(input[2][0] == 9);
  }
  {
    vector<vector<int>> input {{100, 105}, {1, 104}};
    insertion_sort_intervals(input);
    BOOST_TEST(input[0][0] == 1);
    BOOST_TEST(input[1][0] == 100);
  }
  {
    vector<vector<int>> input {
      {89, 90},
      {-10, 20},
      {-50, 0},
      {70, 90},
      {90, 91},
      {90, 95}};
    insertion_sort_intervals(input);
    BOOST_TEST(input[0][0] == -50);
    BOOST_TEST(input[1][0] == -10);
    BOOST_TEST(input[2][0] == 70);
    BOOST_TEST(input[3][0] == 89);
    BOOST_TEST(input[4][0] == 90);
    BOOST_TEST(input[5][0] == 90);
  }
  {
    vector<vector<int>> input {
      {43, 49},
      {9, 12},
      {12, 54},
      {45, 90},
      {91, 93}};
    insertion_sort_intervals(input);
    BOOST_TEST(input[0][0] == 9);
    BOOST_TEST(input[1][0] == 12);
    BOOST_TEST(input[2][0] == 43);
    BOOST_TEST(input[3][0] == 45);
    BOOST_TEST(input[4][0] == 91);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MergeOverlappingIntervalsWorks)
{
  {
    // Video explanation case.
    const vector<vector<int>> input {{1, 2}, {3, 5}, {4, 7}, {6, 8}, {9, 10}};
    const vector<vector<int>> expected {{1, 2}, {3, 8}, {9, 10}};


  }
  {
    // Test case 1
    const vector<vector<int>> input {{1, 2}, {3, 5}, {4, 7}, {6, 8}, {9, 10}};
    const vector<vector<int>> expected {{1, 2}, {3, 8}, {9, 10}};
    const vector<vector<int>> result {merge_overlapping_intervals(input)};

    BOOST_TEST(result.size() == 3);
    BOOST_TEST(result[0][0] == expected[0][0]);
    BOOST_TEST(result[0][1] == expected[0][1]);
    BOOST_TEST(result[0] == expected[0]);
    BOOST_TEST(result[1] == expected[1]);
    BOOST_TEST(result[2] == expected[2]);
  }
  {
    // Test case 2
    const vector<vector<int>> input {{1, 3}, {2, 8}, {9, 10}};
    const vector<vector<int>> expected {{1, 8}, {9, 10}};
    const vector<vector<int>> result {merge_overlapping_intervals(input)};
    BOOST_TEST(result.size() == 2);
    BOOST_TEST(result[0][0] == expected[0][0]);
    BOOST_TEST(result[0][1] == expected[0][1]);
    BOOST_TEST(result[0] == expected[0]);
    BOOST_TEST(result[1] == expected[1]);
  }
  {
    // Test case 5
    const vector<vector<int>> input {{100, 105}, {1, 104}};
    const vector<vector<int>> expected {{1, 105}};
    const vector<vector<int>> result {merge_overlapping_intervals(input)};
    BOOST_TEST(result.size() == 1);
    BOOST_TEST(result[0][0] == expected[0][0]);
    BOOST_TEST(result[0][1] == expected[0][1]);
  }
  {
    // Test case 6
    const vector<vector<int>> input {
      {89, 90},
      {-10, 20},
      {-50, 0},
      {70, 90},
      {90, 91},
      {90, 95}};
    const vector<vector<int>> expected {{-50, 20}, {70, 95}};
    const vector<vector<int>> result {merge_overlapping_intervals(input)};
    BOOST_TEST(result.size() == 2);
    BOOST_TEST(result[0] == expected[0]);
    BOOST_TEST(result[1] == expected[1]);
  }
    // Test case 8
    const vector<vector<int>> input {
      {43, 49},
      {9, 12},
      {12, 54},
      {45, 90},
      {91, 93}};
    const vector<vector<int>> expected {{9, 90}, {91, 93}};
    const vector<vector<int>> result {merge_overlapping_intervals(input)};
    BOOST_TEST(result.size() == 2);
    BOOST_TEST(result[0][0] == expected[0][0]);
    BOOST_TEST(result[1][0] == expected[1][0]);
}

BOOST_AUTO_TEST_SUITE_END() // MergeOverlappingIntervals

BOOST_AUTO_TEST_SUITE_END() // ExpertIo
BOOST_AUTO_TEST_SUITE_END() // Algorithms