#include "Algorithms/LeetCode/EasyProblems.h"

#include <boost/test/unit_test.hpp>
#include <unordered_set>
#include <vector>

using Algorithms::LeetCode::GetMaximumInGeneratedArray;
using Algorithms::LeetCode::TwoSum;
using std::unordered_set;
using std::vector;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(LeetCode)

BOOST_AUTO_TEST_SUITE(TwoSum_0001_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BruteForceGetsTarget)
{
  {
    vector<int> nums {2,7,11,15};
    int target {9};
    unordered_set<int> expected {0, 1};

    const auto result = TwoSum::brute_force(nums, target);

    BOOST_TEST(unordered_set<int>(result.begin(), result.end()) == expected);
  }
  {
    vector<int> nums {3,2,4};
    int target {6};
    unordered_set<int> expected {1, 2};

    const auto result = TwoSum::brute_force(nums, target);

    BOOST_TEST(unordered_set<int>(result.begin(), result.end()) == expected);
  }
  {
    vector<int> nums {3,3};
    int target {6};
    unordered_set<int> expected {0, 1};

    const auto result = TwoSum::brute_force(nums, target);

    BOOST_TEST(unordered_set<int>(result.begin(), result.end()) == expected);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TwoSumWithMapGetsTarget)
{
  {
    vector<int> nums {2,7,11,15};
    int target {9};
    unordered_set<int> expected {0, 1};

    const auto result = TwoSum::two_sum(nums, target);

    BOOST_TEST(unordered_set<int>(result.begin(), result.end()) == expected);
  }
  {
    vector<int> nums {3,2,4};
    int target {6};
    unordered_set<int> expected {1, 2};

    const auto result = TwoSum::two_sum(nums, target);

    BOOST_TEST(unordered_set<int>(result.begin(), result.end()) == expected);
  }
  {
    vector<int> nums {3,3};
    int target {6};
    unordered_set<int> expected {0, 1};

    const auto result = TwoSum::two_sum(nums, target);

    BOOST_TEST(unordered_set<int>(result.begin(), result.end()) == expected);
  }
}

BOOST_AUTO_TEST_SUITE_END() // TwoSum_0001_tests

BOOST_AUTO_TEST_SUITE(GetMaximumInGeneratedArray_1646_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(GetMaximumGeneratedGetsMaximum)
{
  // Example 1
  {
    const int n {7};
    const int expected {3};

    BOOST_TEST(
      GetMaximumInGeneratedArray::get_maximum_generated(n) == expected);
  }
  // Example 2
  {
    const int n {2};
    const int expected {1};
    BOOST_TEST(
      GetMaximumInGeneratedArray::get_maximum_generated(n) == expected);
  }
  // Example 3
  {
    const int n {3};
    const int expected {2};
    BOOST_TEST(
      GetMaximumInGeneratedArray::get_maximum_generated(n) == expected);
  }
}

BOOST_AUTO_TEST_SUITE_END() // GetMaximumInGeneratedArray_1646_tests

BOOST_AUTO_TEST_SUITE_END() // LeetCode
BOOST_AUTO_TEST_SUITE_END() // Algorithms