#include "Algorithms/LeetCode/HardProblems.h"

#include <boost/test/unit_test.hpp>
#include <string>
#include <vector>

// Ordered by Leetcode number.
using Algorithms::LeetCode::MinimumWindowSubstring;
using Algorithms::LeetCode::WaysToEarnPoints;
using Algorithms::LeetCode::MinimumCostToCutStick;
using std::string;
using std::vector;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(LeetCode)

// 76. Minimum Window Substring
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MinimumWindowSubstringWorksWithSlidingWindow)
{
  {
    const string s {"ADOBECODEBANC"};
    const string t {"ABC"};
    const string expected {"BANC"};

    BOOST_TEST(MinimumWindowSubstring::minimum_window(s, t) == expected);
  }
  {
    const string s {"a"};
    const string t {"a"};
    const string expected {"a"};

    BOOST_TEST(MinimumWindowSubstring::minimum_window(s, t) == expected);
  }
  {
    const string s {"a"};
    const string t {"aa"};
    const string expected {""};

    BOOST_TEST(MinimumWindowSubstring::minimum_window(s, t) == expected);
  }  
}

/// 1293. Shortest Path in a Grid with Obstacles Elimination
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ShortestPathInGridWithObstaclesFindShortest)
{
  {
    vector<vector<int>> grid {
      {0, 0, 0},
      {1, 1, 0},
      {0, 0, 0},
      {0, 1, 1},
      {0, 0, 0}};
    int k {1};

    int output {6};
  }

  {
    vector<vector<int>> grid {{0, 1, 1}, {1, 1, 1}, {1, 0, 0}};
    int k {1};

    // We needed to eliminate at least 2 obstables to find such a walk.
    int output {-1};
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MinimumCostToCutStickFindsMinimum)
{
  // Example 1
  {
    const int n {7};
    vector<int> cuts {1, 3, 4, 5};
    const int expected {16};

    BOOST_TEST(
      MinimumCostToCutStick::minimum_cost_to_cut_stick(n, cuts) == expected);
  }
  // Example 2
  {
    const int n {9};
    vector<int> cuts {5, 6, 1, 4, 2};
    const int expected {22};

    BOOST_TEST(
      MinimumCostToCutStick::minimum_cost_to_cut_stick(n, cuts) == expected);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(WaysToEarnPointsReachesTarget)
{
  // Example 1
  {
    const int target {6};
    vector<vector<int>> types {{6,1}, {3,2}, {2,3}};

    const int expected {7};

    BOOST_TEST(WaysToEarnPoints::ways_to_reach_target(target, types), expected);
  }
  // Example 2
  {
    const int target {5};
    vector<vector<int>> types {{50,1}, {50,2}, {50,5}};

    const int expected {4};

    BOOST_TEST(WaysToEarnPoints::ways_to_reach_target(target, types), expected);
  }
  // Example 3
  {
    const int target {18};
    vector<vector<int>> types {{6,1},{3,2},{2,3}};

    const int expected {1};

    BOOST_TEST(WaysToEarnPoints::ways_to_reach_target(target, types), expected);
  }
}

BOOST_AUTO_TEST_SUITE_END() // LeetCode
BOOST_AUTO_TEST_SUITE_END() // Algorithms