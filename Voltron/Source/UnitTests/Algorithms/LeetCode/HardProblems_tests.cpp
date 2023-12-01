#include "Algorithms/LeetCode/HardProblems.h"

#include <boost/test/unit_test.hpp>
#include <vector>

using Algorithms::LeetCode::WaysToEarnPoints;
using std::vector;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(LeetCode)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(WaysToEarnPointsReachesTarget)
{
  // Example 1
  {
    const int target {6};
    const vector<vector<int>> types {{6,1}, {3,2}, {2,3}};

    const int expected {7};
  }
  // Example 2
  {
    const int target {5};
    const vector<vector<int>> types {{50,1}, {50,2}, {50,5}};

    const int expected {4};
  }
  // Example 2
  {
    const int target {18};
    const vector<vector<int>> types {{6,1},{3,2},{2,3}};

    const int expected {1};
  }
}

BOOST_AUTO_TEST_SUITE_END() // LeetCode
BOOST_AUTO_TEST_SUITE_END() // Algorithms