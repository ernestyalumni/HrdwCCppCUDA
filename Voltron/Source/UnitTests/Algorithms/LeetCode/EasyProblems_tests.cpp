#include "Algorithms/LeetCode/EasyProblems.h"

#include <boost/test/unit_test.hpp>
#include <string>
#include <unordered_set>
#include <vector>

using Algorithms::LeetCode::BestTimeToBuyAndSellStock;
using Algorithms::LeetCode::BinarySearch;
using Algorithms::LeetCode::GetMaximumInGeneratedArray;
using Algorithms::LeetCode::TwoSum;
using Algorithms::LeetCode::MergeSortedArray;
using Algorithms::LeetCode::ValidAnagram;
using Algorithms::LeetCode::ValidPalindrome;
using std::string;
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

/// 88. Merge Sorted Array

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MergeSortedArrayWithTwoPointers)
{
  {
    vector<int> nums1 {1,2,3,0,0,0};
    const int m {3};
    vector<int> nums2 {2,5,6};
    const int n {3};

    const vector<int> expected {1,2,2,3,5,6};

    MergeSortedArray::merge(nums1, m, nums2, n);

    BOOST_TEST(nums1 == expected);
  }
  {
    vector<int> nums1 {1};
    const int m {1};
    vector<int> nums2 {};
    const int n {0};
    const vector<int> expected {1};

    MergeSortedArray::merge(nums1, m, nums2, n);

    BOOST_TEST(nums1 == expected);
  }
  {
    vector<int> nums1 {0};
    const int m {0};
    vector<int> nums2 {1};
    const int n {1};
    const vector<int> expected {1};

    MergeSortedArray::merge(nums1, m, nums2, n);

    BOOST_TEST(nums1 == expected);
  }
  // Test Case 9 / 59
  {
    vector<int> nums1 {4,0,0,0,0,0};
    const int m {1};
    vector<int> nums2 {1,2,3,5,6};
    const int n {5};
    const vector<int> expected {1,2,3,4,5,6};

    MergeSortedArray::merge(nums1, m, nums2, n);

    BOOST_TEST(nums1 == expected);
  }  
}

//------------------------------------------------------------------------------
/// 121. Best Time to Buy and Sell Stock
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IterateOnceKeepingTrackOfMinAndMaxWorks)
{
  {
    vector<int> prices {7,1,5,3,6,4};
    const int expected {5};

    BOOST_TEST(BestTimeToBuyAndSellStock::max_profit(prices) == expected);
  }
  {
    vector<int> prices {7,6,4,3,1};
    const int expected {0};
    BOOST_TEST(BestTimeToBuyAndSellStock::max_profit(prices) == expected);
  }
}

//------------------------------------------------------------------------------
/// 125. Valid Palindrome
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ValidPalindromeWorksLinearly)
{
  {
    const string s {"A man, a plan, a canal: Panama"};

    BOOST_TEST(ValidPalindrome::is_palindrome(s));
  }
  {
    const string s {"race a car"};
    BOOST_TEST(!ValidPalindrome::is_palindrome(s));
  }
  {
    const string s {" "};
    BOOST_TEST(ValidPalindrome::is_palindrome(s));
  }
  // 463 / 485 Test case.
  {
    const string s {"0P"};
    BOOST_TEST(!ValidPalindrome::is_palindrome(s));
  }
}

//------------------------------------------------------------------------------
/// 217. Contains Duplicate
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(UnorderedMapFindsDuplicates)
{
  {
    vector<int> nums {1,2,3,1};
    const bool expected {true};

    BOOST_TEST(ContainsDuplicate::contains_duplicate(nums));
  }
  {
    vector<int> nums {1,2,3,4};
    const bool expected {false};
    BOOST_TEST(!ContainsDuplicate::contains_duplicate(nums));
  }
  {
    vector<int> nums {1,1,1,3,3,4,3,2,4,2};
    const bool expected {true};
    BOOST_TEST(ContainsDuplicate::contains_duplicate(nums));
  }
}

//------------------------------------------------------------------------------
/// 242. Valid Anagram
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IsAnagramWorksWithUnorderedMap)
{
  {
    const string s {"anagram"};
    const string t {"nagaram"};

    BOOST_TEST(ValidAnagram::is_anagram(s, t));
  }
  {
    const string s {"rat"};
    const string t {"car"};

    BOOST_TEST(!ValidAnagram::is_anagram(s, t));
  }
}

//------------------------------------------------------------------------------
/// 704. Binary Search
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BinarySearchReturnsIndexOrNegative1)
{
  {
    vector<int> nums {-1,0,3,5,9,12};
    const int target {9};
    const int expected {4};
    BOOST_TEST(BinarySearch::search(nums, target) == expected);
  }
  {
    vector<int> nums {-1,0,3,5,9,12};
    const int target {2};
    const int expected {-1};
    BOOST_TEST(BinarySearch::search(nums, target) == expected);
  }
}

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