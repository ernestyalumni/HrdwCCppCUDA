#include "Algorithms/LeetCode/MediumProblems.h"

#include <boost/test/unit_test.hpp>
#include <string>
#include <vector>

using Algorithms::LeetCode::LongestPalindrome;
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

BOOST_AUTO_TEST_SUITE_END() // LeetCode
BOOST_AUTO_TEST_SUITE_END() // Algorithms