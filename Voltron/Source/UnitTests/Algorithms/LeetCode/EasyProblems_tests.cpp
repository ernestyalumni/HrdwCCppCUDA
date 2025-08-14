#include "Algorithms/LeetCode/EasyProblems.h"
#include "DataStructures/BinaryTrees.h"

#include <boost/test/unit_test.hpp>
#include <string>
#include <unordered_set>
#include <vector>

// TODO: It may not be necessary to alias the classes because Boost's test
// brings the namespace in.

using Algorithms::LeetCode::BestTimeToBuyAndSellStock;
using Algorithms::LeetCode::BinarySearch;
using Algorithms::LeetCode::GetMaximumInGeneratedArray;
using Algorithms::LeetCode::TwoSum;
using Algorithms::LeetCode::MergeSortedArray;
using Algorithms::LeetCode::ValidAnagram;
using Algorithms::LeetCode::ValidPalindrome;
using DataStructures::BinaryTrees::TreeNode;
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

//------------------------------------------------------------------------------
/// 20. Valid Parentheses
/// https://leetcode.com/problems/valid-parentheses/description/
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ValidParenthesesReturnsValidParentheses)
{
  // https://leetcode.com/problems/valid-parentheses/description/
  {
    string s {"()"};

    BOOST_TEST(ValidParentheses::is_valid(s));
  }
  {
    string s {"()[]{}"};
    BOOST_TEST(ValidParentheses::is_valid(s));
  }
  {
    string s {"(]"};
    BOOST_TEST(!ValidParentheses::is_valid(s));
  }
  {
    string s {"([])"};
    BOOST_TEST(ValidParentheses::is_valid(s));
  }
  // 6 / 99 testcases passed
  {
    string s {"["};
    BOOST_TEST(!ValidParentheses::is_valid(s));    
  }

  // EY 20240921
  {
    string s {"}"};
    BOOST_TEST(!ValidParentheses::is_valid(s));    
  }
}

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
/// 104. Maximum Depth of Binary Tree
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(GetMaximumDepthOfBinaryTreeByRecursion)
{
  {
    TreeNode example_root {3};
    TreeNode c1l2 {9};
    TreeNode c1r3 {20};
    TreeNode c3l4 {15};
    TreeNode c3r5 {7};

    example_root.left_ = &c1l2;
    example_root.right_ = &c1r3;
    c1r3.left_ = &c3l4;
    c1r3.right_ = &c3r5;

    TreeNode* root {&example_root};

    const int expected {3};

    BOOST_TEST(MaximumDepthOfBinaryTree::max_depth_recursive(root) == expected);
  }
  {
    TreeNode example_root {1};
    TreeNode c1r2 {2};
    example_root.right_ = &c1r2;
    TreeNode* root {&example_root};
    const int expected {2};

    BOOST_TEST(MaximumDepthOfBinaryTree::max_depth_recursive(root) == expected);
  }
  // Test case 17 / 39
  {
    TreeNode example_root {1};
    TreeNode c1l2 {2};
    TreeNode c1r3 {3};
    TreeNode c2l4 {4};
    TreeNode c3r5 {5};

    example_root.left_ = &c1l2;
    example_root.right_ = &c1r3;
    c1l2.left_ = &c2l4;
    c1r3.right_ = &c3r5;

    TreeNode* root {&example_root};

    const int expected {3};

    BOOST_TEST(MaximumDepthOfBinaryTree::max_depth_recursive(root) == expected);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(GetMaximumDepthOfBinaryTreeIteratively)
{
  {
    TreeNode example_root {3};
    TreeNode c1l2 {9};
    TreeNode c1r3 {20};
    TreeNode c3l4 {15};
    TreeNode c3r5 {7};

    example_root.left_ = &c1l2;
    example_root.right_ = &c1r3;
    c1r3.left_ = &c3l4;
    c1r3.right_ = &c3r5;

    TreeNode* root {&example_root};

    const int expected {3};

    BOOST_TEST(MaximumDepthOfBinaryTree::max_depth_iterative(root) == expected);
  }
  {
    TreeNode example_root {1};
    TreeNode c1r2 {2};
    example_root.right_ = &c1r2;
    TreeNode* root {&example_root};
    const int expected {2};

    BOOST_TEST(MaximumDepthOfBinaryTree::max_depth_iterative(root) == expected);
  }
  // Test case 17 / 39
  {
    TreeNode example_root {1};
    TreeNode c1l2 {2};
    TreeNode c1r3 {3};
    TreeNode c2l4 {4};
    TreeNode c3r5 {5};

    example_root.left_ = &c1l2;
    example_root.right_ = &c1r3;
    c1l2.left_ = &c2l4;
    c1r3.right_ = &c3r5;

    TreeNode* root {&example_root};

    const int expected {3};

    BOOST_TEST(MaximumDepthOfBinaryTree::max_depth_iterative(root) == expected);
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
/// 136. Single Number
/// Given a non-empty array of integers (nums), every element appears twice
/// except for one. Find that single one.
/// https://leetcode.com/problems/single-number/description/
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SingleNumberWithSet)
{
  // Example 1
  vector<int> nums {2,2,1};
  int expected_output {1};
  BOOST_TEST(SingleNumber::single_number_with_set(nums) == expected_output);

  nums = {4,1,2,1,2};
  expected_output = 4;
  BOOST_TEST(SingleNumber::single_number_with_set(nums) == expected_output);

  nums = {1};
  expected_output = 1;
  BOOST_TEST(SingleNumber::single_number_with_set(nums) == expected_output);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SingleNumberWithXOR)
{
  // Example 1
  vector<int> nums {2,2,1};
  int expected_output {1};
  BOOST_TEST(SingleNumber::single_number_xor(nums) == expected_output);

  nums = {4,1,2,1,2};
  expected_output = 4;
  BOOST_TEST(SingleNumber::single_number_xor(nums) == expected_output);

  nums = {1};
  expected_output = 1;
  BOOST_TEST(SingleNumber::single_number_xor(nums) == expected_output);
}

//------------------------------------------------------------------------------
/// 169. Majority Element
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MajorityElementWithMapGetsMajorityElement)
{
  {
    vector<int> nums {3, 2, 3};
    const int expected {3};

    BOOST_CHECK_EQUAL(
      MajorityElement::majority_element_with_map(nums),
      expected);
  }
  {
    vector<int> nums {2,2,1,1,1,2,2}; 
    const int expected {2};
    BOOST_CHECK_EQUAL(
      MajorityElement::majority_element_with_map(nums),
      expected);
  }
  {
    vector<int> nums {1};
    const int expected {1};
    BOOST_CHECK_EQUAL(
      MajorityElement::majority_element_with_map(nums),
      expected);
  }
}

//------------------------------------------------------------------------------
/// 190. Reverse Bits
/// https://leetcode.com/problems/reverse-bits/description/
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ReverseBitsLoopThrough)
{
  // Example 1
  int n {43261596};
  int expected_output {964176192};
  BOOST_TEST(ReverseBits::reverse_bits_loop_through(n) == expected_output);

  // Example 2
  n = 2147483644;
  expected_output = 1073741822;
  BOOST_TEST(ReverseBits::reverse_bits_loop_through(n) == expected_output);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ReverseBitsGetAndShiftLSB)
{
  // Example 1
  int n {43261596};
  int expected_output {964176192};
  BOOST_TEST(ReverseBits::reverse_bits_get_and_shift_lsb(n) == expected_output);

  // Example 2
  n = 2147483644;
  expected_output = 1073741822;
  BOOST_TEST(ReverseBits::reverse_bits_get_and_shift_lsb(n) == expected_output);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ReverseBitsSwapHalves)
{
  // Example 1
  int n {43261596};
  int expected_output {964176192};
  BOOST_TEST(ReverseBits::reverse_bits_swap_halves(n) == expected_output);

  // Example 2
  n = 2147483644;
  expected_output = 1073741822;
  BOOST_TEST(ReverseBits::reverse_bits_swap_halves(n) == expected_output);
}

//------------------------------------------------------------------------------
/// 191. Number of 1 Bits
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(NumberOf1BitsLoopAllBits)
{
  // Example 1
  int n {11};
  int expected_output {3};
  BOOST_TEST(NumberOf1Bits::hamming_weight_loop_all_bits(n) == expected_output);

  // Example 2
  n = 128;
  expected_output = 1;
  BOOST_TEST(NumberOf1Bits::hamming_weight_loop_all_bits(n) == expected_output);

  // Example 
  n = 2147483645;
  expected_output = 30;
  BOOST_TEST(NumberOf1Bits::hamming_weight_loop_all_bits(n) == expected_output);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(NumberOf1BitsKernighanTrick)
{
  // Example 1
  int n {11};
  int expected_output {3};
  BOOST_TEST(
    NumberOf1Bits::hamming_weight_kernighan_trick(n) == expected_output);

  // Example 2
  n = 128;
  expected_output = 1;
  BOOST_TEST(
    NumberOf1Bits::hamming_weight_kernighan_trick(n) == expected_output);

  // Example 
  n = 2147483645;
  expected_output = 30;
  BOOST_TEST(
    NumberOf1Bits::hamming_weight_kernighan_trick(n) == expected_output);
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
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SortingFirstFindsDuplicates)
{
  {
    vector<int> nums {1,2,3,1};
    const bool expected {true};

    BOOST_TEST(ContainsDuplicate::sort_first(nums));
  }
  {
    vector<int> nums {1,2,3,4};
    const bool expected {false};
    BOOST_TEST(!ContainsDuplicate::sort_first(nums));
  }
  {
    vector<int> nums {1,1,1,3,3,4,3,2,4,2};
    const bool expected {true};
    BOOST_TEST(ContainsDuplicate::sort_first(nums));
  }
}

//------------------------------------------------------------------------------
/// 226. Invert Binary Tree
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(RecursionInvertsBinaryTree)
{
  {
    TreeNode example_root {4};
    TreeNode c1l2 {2};
    TreeNode c1r3 {7};
    TreeNode c2l4 {1};
    TreeNode c2r5 {3};
    TreeNode c3l6 {6};
    TreeNode c3r7 {9};

    example_root.left_ = &c1l2;
    example_root.right_ = &c1r3;
    c1l2.left_ = &c2l4;
    c1l2.right_ = &c2r5;
    c1r3.left_ = &c3l6;
    c1r3.right_ = &c3r7;

    TreeNode* root {&example_root};

    const TreeNode* output {InvertBinaryTree::invert_tree_recursive(root)};

    BOOST_TEST(output->value_ == 4);
    BOOST_TEST(output->left_->value_ == 7);
    BOOST_TEST(output->right_->value_ == 2);
    BOOST_TEST(output->left_->left_->value_ == 9);
    BOOST_TEST(output->left_->right_->value_ == 6);
  }
  {
    TreeNode example_root {2};
    TreeNode c1l2 {1};
    TreeNode c1r3 {3};
    example_root.left_ = &c1l2;
    example_root.right_ = &c1r3;

    TreeNode* root {&example_root};

    const TreeNode* output {InvertBinaryTree::invert_tree_recursive(root)};

    BOOST_TEST(output->value_ == 2);
    BOOST_TEST(output->left_->value_ == 3);
    BOOST_TEST(output->right_->value_ == 1);
  }
  {
    TreeNode* root {nullptr};
    const TreeNode* output {InvertBinaryTree::invert_tree_recursive(root)};
    BOOST_TEST(output == nullptr);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IterativeStackInvertsBinaryTree)
{
  {
    TreeNode example_root {4};
    TreeNode c1l2 {2};
    TreeNode c1r3 {7};
    TreeNode c2l4 {1};
    TreeNode c2r5 {3};
    TreeNode c3l6 {6};
    TreeNode c3r7 {9};

    example_root.left_ = &c1l2;
    example_root.right_ = &c1r3;
    c1l2.left_ = &c2l4;
    c1l2.right_ = &c2r5;
    c1r3.left_ = &c3l6;
    c1r3.right_ = &c3r7;

    TreeNode* root {&example_root};

    const TreeNode* output {InvertBinaryTree::invert_tree_iterative(root)};

    BOOST_TEST(output->value_ == 4);
    BOOST_TEST(output->left_->value_ == 7);
    BOOST_TEST(output->right_->value_ == 2);
    BOOST_TEST(output->left_->left_->value_ == 9);
    BOOST_TEST(output->left_->right_->value_ == 6);
  }
  {
    TreeNode example_root {2};
    TreeNode c1l2 {1};
    TreeNode c1r3 {3};
    example_root.left_ = &c1l2;
    example_root.right_ = &c1r3;

    TreeNode* root {&example_root};

    const TreeNode* output {InvertBinaryTree::invert_tree_iterative(root)};

    BOOST_TEST(output->value_ == 2);
    BOOST_TEST(output->left_->value_ == 3);
    BOOST_TEST(output->right_->value_ == 1);
  }
  {
    TreeNode* root {nullptr};
    const TreeNode* output {InvertBinaryTree::invert_tree_iterative(root)};
    BOOST_TEST(output == nullptr);
  }
}

//------------------------------------------------------------------------------
/// 231. Power of Two
/// https://leetcode.com/problems/power-of-two/description/
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PowerOfTwoAnd)
{
  // Example 1
  int n {1};
  BOOST_TEST(PowerOfTwo::is_power_of_two_and(n));

  // Case 2
  n = 16;
  BOOST_TEST(PowerOfTwo::is_power_of_two_and(n));

  // Case 3
  n = 3;
  BOOST_TEST(!PowerOfTwo::is_power_of_two_and(n));

  // Case 595
  n = 6;
  BOOST_TEST(!PowerOfTwo::is_power_of_two_and(n));

  // Case 1110
  n = 0;
  BOOST_TEST(!PowerOfTwo::is_power_of_two_and(n));
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
/// 268. Missing Number
/// https://leetcode.com/problems/missing-number/description/
/// The idea is to use XOR.
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MissingNumberXOR)
{
  // Example 1
  vector<int> nums {3,0,1};
  int expected_output {2};
  BOOST_TEST(MissingNumber::missing_number_xor(nums) == expected_output);

  // Example 2
  nums = {0,1};
  expected_output = 2;
  BOOST_TEST(MissingNumber::missing_number_xor(nums) == expected_output);

  // Example 3
  nums = {9,6,4,2,3,5,7,0,1};
  expected_output = 8;
  BOOST_TEST(MissingNumber::missing_number_xor(nums) == expected_output);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MissingNumberSumFormula)
{
  // Example 1
  vector<int> nums {3,0,1};
  int expected_output {2};
  BOOST_TEST(
    MissingNumber::missing_number_sum_formula(nums) == expected_output);

  // Example 2
  nums = {0,1};
  expected_output = 2;
  BOOST_TEST(
    MissingNumber::missing_number_sum_formula(nums) == expected_output);

  // Example 3
  nums = {9,6,4,2,3,5,7,0,1};
  expected_output = 8;
  BOOST_TEST(
    MissingNumber::missing_number_sum_formula(nums) == expected_output);
}

//------------------------------------------------------------------------------
/// 338. Counting Bits
/// https://leetcode.com/problems/counting-bits/description/
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CountingBitsMemoization)
{
  // Example 1
  int n {2};
  vector<int> expected_output {0, 1, 1};

  BOOST_TEST(CountingBits::count_bits_memoization(n) == expected_output);

  // Example 2
  n = 5;
  expected_output = {0 ,1, 1, 2, 1, 2};
  BOOST_TEST(CountingBits::count_bits_memoization(n) == expected_output);
}

//------------------------------------------------------------------------------
/// 405. Convert a Number to Hexadecimal
/// https://leetcode.com/problems/convert-a-number-to-hexadecimal/description/
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConvertANumberToHexadecimalConvertsToHexadecimal)
{
  // Example 1
  int num {26};
  string expected_output {"1a"};

  BOOST_TEST(ConvertToHexadecimal::to_hex(num) == expected_output);

  // Example 2
  num = -1;
  expected_output = "ffffffff";
  BOOST_TEST(ConvertToHexadecimal::to_hex(num) == expected_output);

  // Case 100
  num = 0;
  expected_output = "0";
  BOOST_TEST(ConvertToHexadecimal::to_hex(num) == expected_output);
}

//------------------------------------------------------------------------------
/// 461. Hamming Distance
/// https://leetcode.com/problems/hamming-distance/description/
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(HammingDistanceCounts)
{
  // Example 1
  int x {1};
  int y {4};
  int expected_output {2};
  BOOST_TEST(HammingDistance::hamming_distance(x, y) == expected_output);

  // Example 2
  x = 3;
  y = 1;
  expected_output = 1;
  BOOST_TEST(HammingDistance::hamming_distance(x, y) == expected_output);
}

//------------------------------------------------------------------------------
/// 697. Degree of an Arrays
//------------------------------------------------------------------------------

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

//------------------------------------------------------------------------------
/// 733. Flood Fill
//------------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE(FloodFill_0733_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(RecursiveDepthFirstSearchFills)
{
  {
    vector<vector<int>> image {{1,1,1},{1,1,0},{1,0,1}};
    const int sr {1};
    const int sc {1};
    const int color {2};
    const vector<vector<int>> expected {{2,2,2},{2,2,0},{2,0,1}};

    BOOST_TEST(FloodFill::flood_fill(image, sr, sc, color) == expected);
  }
  {
    vector<vector<int>> image {{0,0,0},{0,0,0}};
    const int sr {0};
    const int sc {0};
    const int color {0};
    const vector<vector<int>> expected {{0,0,0},{0,0,0}};

    BOOST_TEST(FloodFill::flood_fill(image, sr, sc, color) == expected);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BreadthFirstSearchWithQueueFills)
{
  {
    vector<vector<int>> image {{1,1,1},{1,1,0},{1,0,1}};
    const int sr {1};
    const int sc {1};
    const int color {2};
    const vector<vector<int>> expected {{2,2,2},{2,2,0},{2,0,1}};

    BOOST_TEST(FloodFill::flood_fill_with_queue(image, sr, sc, color) ==
      expected);
  }
  {
    vector<vector<int>> image {{0,0,0},{0,0,0}};
    const int sr {0};
    const int sc {0};
    const int color {0};
    const vector<vector<int>> expected {{0,0,0},{0,0,0}};

    BOOST_TEST(FloodFill::flood_fill_with_queue(image, sr, sc, color) ==
      expected);
  }
}

BOOST_AUTO_TEST_SUITE_END() // FloodFill_0733_tests

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