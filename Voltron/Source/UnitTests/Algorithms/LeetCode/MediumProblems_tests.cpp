#include "Algorithms/LeetCode/MediumProblems.h"

#include <boost/test/unit_test.hpp>
#include <string>
#include <vector>

using Algorithms::LeetCode::LongestPalindrome;
using Algorithms::LeetCode::MinimumNumberOfCoinsForFruits;
using std::string;
using std::vector;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(LeetCode)

BOOST_AUTO_TEST_SUITE(LongestPalindrome_5_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BruteForceFindsLongest)
{
  // Example 1
  {
    const string s {"babad"};
    const string expected {"bab"};

    const string result {LongestPalindrome::brute_force(s)};

    BOOST_TEST(result.size(), expected.size());
    BOOST_TEST(result == expected);
  }
  // Example 2
  {
    const string s {"cbbd"};
    const string expected {"bb"};

    const string result {LongestPalindrome::brute_force(s)};

    BOOST_TEST(result.size(), expected.size());
    BOOST_TEST(result == expected);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ExpandAroundCenterFindsLongest)
{
  // Example 1
  {
    const string s {"babad"};
    const string expected {"bab"};

    const string result {LongestPalindrome::expand_around_center(s)};

    BOOST_TEST(result.size(), expected.size());
    BOOST_TEST(result == expected);
  }
  // Example 2
  {
    const string s {"cbbd"};
    const string expected {"bb"};

    const string result {LongestPalindrome::expand_around_center(s)};

    BOOST_TEST(result.size(), expected.size());
    BOOST_TEST(result == expected);
  }
  // Test Case 124?
  {
    const string s {"ccc"};
    const string expected {"ccc"};

    const string result {LongestPalindrome::expand_around_center(s)};

    BOOST_TEST(result.size(), expected.size());
    BOOST_TEST(result == expected);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DynamicProgrammingFindsLongest)
{
  // Example 1
  {
    const string s {"babad"};
    const string expected {"bab"};

    const string result {LongestPalindrome::with_dynamic_programming(s)};

    BOOST_TEST(result.size(), expected.size());
    BOOST_TEST(result == expected);
  }
  // Example 2
  {
    const string s {"cbbd"};
    const string expected {"bb"};

    const string result {LongestPalindrome::with_dynamic_programming(s)};

    BOOST_TEST(result.size(), expected.size());
    BOOST_TEST(result == expected);
  }
  // Test Case 124?
  {
    const string s {"ccc"};
    const string expected {"ccc"};

    const string result {LongestPalindrome::with_dynamic_programming(s)};

    BOOST_TEST(result.size(), expected.size());
    BOOST_TEST(result == expected);
  }
}

BOOST_AUTO_TEST_SUITE_END() // LongestPalindrome_5_tests

BOOST_AUTO_TEST_SUITE(NumberOfProvinces_547_tests)

const vector<vector<int>> example_1 {{1, 1, 0}, {1, 1, 0}, {0, 0, 1}};
const vector<vector<int>> example_2 {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

const vector<vector<int>> test_case_66 {
  {1,0,0,1},
  {0,1,1,0},
  {0,1,1,1},
  {1,0,1,1}};

//------------------------------------------------------------------------------
/// \url https://leetcode.com/problems/number-of-provinces/
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateGivenExamples)
{
  vector<vector<int>> example_1_input {example_1};
  vector<vector<int>> example_2_input {example_2};

  const int result_1 {
    Algorithms::LeetCode::find_number_of_provinces(example_1_input)};

  BOOST_TEST(result_1 == 2);

  const int result_2 {
    Algorithms::LeetCode::find_number_of_provinces(example_2_input)};

  BOOST_TEST(result_2 == 3);
}

BOOST_AUTO_TEST_SUITE_END() // NumberOfProvinces_547_tests

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MinimumNumberOfCoinsForFruitsFindsMinimum)
{
  // Example 1
  {
    vector<int> prices {3, 1, 2};
    const int expected {4};

    BOOST_TEST(
      MinimumNumberOfCoinsForFruits::minimum_coins(prices) == expected);
  }
  // Example 2
  {
    vector<int> prices {1, 10, 1, 1};
    const int expected {2};

    BOOST_TEST(
      MinimumNumberOfCoinsForFruits::minimum_coins(prices) == expected);
  }
}

BOOST_AUTO_TEST_SUITE_END() // LeetCode
BOOST_AUTO_TEST_SUITE_END() // Algorithms