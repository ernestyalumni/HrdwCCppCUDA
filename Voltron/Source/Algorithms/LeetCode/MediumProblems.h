#ifndef ALGORITHMS_LEETCODE_MEDIUM_PROBLEMS_H
#define ALGORITHMS_LEETCODE_MEDIUM_PROBLEMS_H

#include "DataStructures/BinaryTrees.h"

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

namespace Algorithms
{
namespace LeetCode
{

//------------------------------------------------------------------------------
/// \name 3. Longest Substring Without Repeating Characters
/// Constraints:
/// 0 <= s.length <= 5 * 10^4
/// s consists of English letters, digits, symbols, and spaces
//------------------------------------------------------------------------------
class LongestSubstringWithoutRepeating
{
  public:

    //--------------------------------------------------------------------------
    /// A substring is a contiguous block of characters so a sliding window can
    /// keep track of a set of contiguous elements and can update the state of
    /// unique characters as we move along the string.
    //--------------------------------------------------------------------------
    static int length_of_longest_substring(std::string s);

    static int concise_length_of_longest_substring(std::string s);
};

//------------------------------------------------------------------------------
/// \name 5. Longest Palindromic Substring
/// \url https://leetcode.com/problems/longest-palindromic-substring/
/// \brief Given a string s, return the longest palindromic substring in s.
//------------------------------------------------------------------------------
class LongestPalindrome
{
  public:

    //--------------------------------------------------------------------------
    /// https://leetcode.com/problems/longest-palindromic-substring/solutions/4212564/beats-96-49-5-different-approaches-brute-force-eac-dp-ma-recursion/
    /// \details Brute force solution is to check every substring. Pick every
    /// possible starting position. For each starting position, pick every
    /// possible end position.
    /// N^2 possible substrings. Verifying each substring takes O(N) time.
    /// O(N^3) time complexity.
    //--------------------------------------------------------------------------
    static std::string brute_force(std::string s);

    // O(N) time complexity.
    static bool is_palindrome(const std::string& s);

    static std::string expand_around_center(std::string s);

    // O(N) time complexity.
    // Assumes that the letters that are included within given inputs left, and
    // right are a palindrome already.
    static std::tuple<int, int> expand_from_center(
      const std::string& s,
      const int left,
      const int right);

    static std::string with_dynamic_programming(std::string s);
    
    static std::string longest_palindrome(std::string s);

    static void find_longest_palindrome(
      int start_index,
      int current_length,
      int& maximum_length,
      int& longest_palindrome_index,
      const std::string& s,
      const int N);

    static inline bool is_outer_letters_palindrome(
      const std::size_t start_index,
      const std::size_t current_length,
      const std::string& s)
    {
      return s[start_index] == s[start_index + current_length - 1];
    }
};

//------------------------------------------------------------------------------
/// \name 11. Container With Most Water
/// \details Greedy (tag)
/// \url https://leetcode.com/problems/container-with-most-water/
/// Constraints:
/// 2 <= n <= 10^5 where n == height.length.
/// 0 <= height[i] <= 10^4
///
/// Key ideas: Use 2 (index) pointers. Update max area at each step.
//------------------------------------------------------------------------------
class ContainerWithMostWater
{
  public:

    // O(N^2) time complexity (loop through all possible intervals).
    static int brute_force(std::vector<int>& height);

    // Greedy Algorithm.
    int maximum_area(std::vector<int>& height);
};

//------------------------------------------------------------------------------
/// 15. 3Sum
/// Key idea: Sort the array first. Avoid duplicates by iterating past them.
/// Iterate through every element. For each element, use the two pointer method
/// from the left and right.
/// *Sort the array first.*
//------------------------------------------------------------------------------
class ThreeSum
{
  public:

    static std::vector<std::vector<int>> three_sum(std::vector<int>& nums);
};

//------------------------------------------------------------------------------
/// 33. Search in Rotated Sorted Array
/// Given array nums after the possible rotation and an integer target, return
/// index of target if it's in nums, or -1 if it's not in nums.
/// Constraints
/// 1 <= nums.length <= 5000
///
/// Still do a binary search, but we check both sides of the midpoint if there's
/// a pivot or not (then it's sorted).
//------------------------------------------------------------------------------
class SearchInRotatedSortedArray
{
  public:

    static int search(std::vector<int>& nums, int target);
};

//------------------------------------------------------------------------------
/// 48. Rotate Image
/// Constraints:
/// * n == matrix.length == matrix[i].length
/// * 1 <= n <= 20
//------------------------------------------------------------------------------
class RotateImage
{
  public:

    static void rotate(std::vector<std::vector<int>>& matrix);
};

//------------------------------------------------------------------------------
/// 53. Maximum Subarray
/// Constraints:
/// 1 <= nums.length <= 10^5
/// -10^4 <= nums[i] <= 10^4
/// Consider Kadane's algorithm, and consider both a local and global maxima.
//------------------------------------------------------------------------------
class MaximumSubarray
{
  public:

    //--------------------------------------------------------------------------
    /// Consider 2 single int values, local maximum, global maximum to track.
    //--------------------------------------------------------------------------
    static int max_subarray(std::vector<int>& nums);
};

//------------------------------------------------------------------------------
/// 54. Spiral Matrix
/// Constraints: 1 <= m, n <= 10
/// Key idea: Move the "boundaries" or the "paths" the spiral moves on along as
/// we spiral around.
//------------------------------------------------------------------------------
class SpiralMatrix
{
  public:
    static std::vector<int> spiral_order(std::vector<std::vector<int>>& matrix);
};

//------------------------------------------------------------------------------
/// 56. Merge Intervals
/// Constraints 1 <= intervals.length <= 10^4
/// intervals[i].length == 2
/// 0 <= start_i <= end_i <= 10^4
/// Key ideas: sort elements of the array first.
//------------------------------------------------------------------------------
class MergeIntervals
{
  public:

    static std::vector<std::vector<int>> merge(
      std::vector<std::vector<int>>& intervals);
};

//------------------------------------------------------------------------------
/// 73. Set Matrix Zeroes
//------------------------------------------------------------------------------
class SetMatrixZeroes
{
  public:

    static void brute_force(std::vector<std::vector<int>>& matrix);
    static void set_zeroes(std::vector<std::vector<int>>& matrix);
};

//------------------------------------------------------------------------------
/// 74. Search a 2D Matrix
/// Constraints:
/// m == matrix.length
/// n == matrix[0].length
/// 1 <= m, n <= 200
//------------------------------------------------------------------------------
class SearchA2DMatrix
{
  public:
    static bool search_matrix(
      std::vector<std::vector<int>>& matrix,
      int target);
};

//------------------------------------------------------------------------------
/// \name 75. Sort Colors
/// Constraints
/// * n == nums.length
/// * 1 <= n <= 300
/// * nums[i] is either 0, 1, or 2.
///
/// In-place sorting suggests manipulating array by swapping elements, which is
/// a common operation in 2-pointer techniques.
//------------------------------------------------------------------------------
class SortColors
{
  public:

    static void sort_colors(std::vector<int>& nums);
  
    //--------------------------------------------------------------------------
    /// This is a variant of the Dutch National Flag algorithm with 3 pointers.
    //--------------------------------------------------------------------------
    static void one_pass(std::vector<int>& nums);
};

//------------------------------------------------------------------------------
/// 102. Binary Tree Level Order Traversal
/// Key idea: Use a queue for level order traversal. Use size of queue to get
/// size of each level, so that we know how many to pop out.
//------------------------------------------------------------------------------

class BinaryTreeLevelOrderTraversal
{
  public:

    using TreeNode = DataStructures::BinaryTrees::TreeNode;

    static std::vector<std::vector<int>> level_order_iterative(TreeNode* root);

    static std::vector<std::vector<int>> level_order_recursive(TreeNode* root);
};

//------------------------------------------------------------------------------
/// 152. Maximum Product Subarray
//------------------------------------------------------------------------------

class MaximumProductSubarray
{
  public:

    static int max_product(std::vector<int>& nums);
};

//------------------------------------------------------------------------------
/// 153. Find Minimum in Rotated Sorted Array
/// Key idea: Still use binary search, but modified, searching or considering on
/// the side (left side or right side of the midpoint) that's ascending order.
//------------------------------------------------------------------------------
class FindMinimumInRotatedSortedArray
{
  public:

    static int find_min(std::vector<int>& nums);
};

//------------------------------------------------------------------------------
/// \name 209. Minimum Size Subarray Sum
/// We're given an array of positive integers (that are non-zero).
/// A subarray is a contiguous non-empty sequence of elements within an array.
/// \url https://leetcode.com/problems/minimum-size-subarray-sum/
/// Constraints:
/// 1 <= target <= 10^9
/// 1 <= nums.length <= 10^5
/// 1 <= nums[i] <= 10^4
//------------------------------------------------------------------------------
class MinimumSizeSubarraySum
{
  public:

    /// @brief 
    /// @param target 
    /// @param nums 
    /// Sliding window technique.
    /// @return 
    static int minimum_subarray_length(int target, std::vector<int>& nums);
};

//------------------------------------------------------------------------------
/// 215. Product of Array Except self
//------------------------------------------------------------------------------
class KthLargestElementInAnArray
{
  public:

    static int brute_force(std::vector<int>& nums, int k);
    static int find_kth_largest(std::vector<int>& nums, int k);
};

//------------------------------------------------------------------------------
/// 235. Lowest Common Ancestor of a Binary Search Tree
/// Constraints:
/// All Node.val are unique.
/// p != q
/// p and q will exist in the BST.
/// Key idea: Use recursion. Use property of BST.
//------------------------------------------------------------------------------
class LowestCommonAncestorOfABinarySearchTree
{
  public:

    using TreeNode = DataStructures::BinaryTrees::TreeNode;

    static TreeNode* lowest_common_ancestor_recursive(
      TreeNode* root,
      TreeNode* p,
      TreeNode* q);

    static TreeNode* lowest_common_ancestor_iterative(
      TreeNode* root,
      TreeNode* p,
      TreeNode* q);
};

//------------------------------------------------------------------------------
/// 238. Product of Array Except self
/// Key ideas: precomputation. We want for each i, p[i] = \Prod_{j = 0}^{N-1}
/// x[j] such that j != i. Split this product up into p[i] = \Prod_{j= 0}^{i-1}
/// x[j] * \Prod_{j=i+1}^{N - 1} x[j] = L * R where
/// L are "left products" and R are "right products"
//------------------------------------------------------------------------------
class ProductOfArrayExceptSelf
{
  public:

    static std::vector<int> brute_force(std::vector<int>& nums);
    static std::vector<int> product_except_self(std::vector<int>& nums);
};

//------------------------------------------------------------------------------
/// \name 357. Count Numbers With Unique Digits
/// \url https://leetcode.com/problems/count-numbers-with-unique-digits/
/// Given integer n, return count of all numbers with unique digits, x, where
/// 0 <= x < 10^n.
//------------------------------------------------------------------------------
class CountNumbersWithUniqueDigits
{
  public:

    static int count_numbers_with_unique_digits(int n);
};

//------------------------------------------------------------------------------
/// 378. Kth Smallest Element in a Sorted Matrix
//------------------------------------------------------------------------------
class KthSmallestElementInASortedMatrix
{
  public:

    // https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/solutions/3844831/c-brute/
    static int brute_force(std::vector<std::vector<int>>& matrix, int k);

    static int kth_smallest(std::vector<std::vector<int>>& matrix, int k);
};

//------------------------------------------------------------------------------
/// 424. Longest Repeating Character Replacement
/// Key idea: Sliding window - maintain a sliding window that can contain the
/// longest substring with replacement.
//------------------------------------------------------------------------------
class LongestRepeatingCharacterReplacement
{
  public:

    static int character_replacement(std::string s, int k);
};

//------------------------------------------------------------------------------
/// 435. Non-overlapping intervals
/// Key ideas: sort array.
//------------------------------------------------------------------------------
class NonOverlappingIntervals
{
  public:

    static int erase_overlap_intervals(
      std::vector<std::vector<int>>& intervals);
};

//------------------------------------------------------------------------------
/// 438. Find all anagrams in a string
/// Constraints:
/// s and p consist of lowercase English letters.
/// Key ideas: Unordered map to track what unique characters have been seen.
/// Use 2 hash maps (or arrays) to count characters in p.
//------------------------------------------------------------------------------
class FindAllAnagramsInAString
{
  public:

    static std::vector<int> find_anagrams(std::string s, std::string p);
};

//------------------------------------------------------------------------------
/// 516. Longest Palindromic Subsequence
/// A subsequence is a sequence that can be derived from another sequence by
/// deleting some or no elements without changing the order of the remaining
/// elements.
/// Constraints:
/// 1 <= s.length <= 1000
/// s consists only of lowercase English letters.
/// Key ideas: dynamic programming.
//------------------------------------------------------------------------------
class LongestPalindromicSubsequence
{
  public:

    static int longest_palindrome_subsequence(std::string s);
};

//------------------------------------------------------------------------------
/// \name 547. Number of Provinces
/// \url https://leetcode.com/problems/number-of-provinces/
//------------------------------------------------------------------------------
int find_number_of_provinces(std::vector<std::vector<int>>& is_connected);

//------------------------------------------------------------------------------
/// \name 647. Palindromic Substrings
/// Constraints
/// 1 <= s.length <= 1000
/// s consists of lowercase English letters.
//------------------------------------------------------------------------------
class PalindromicSubstrings
{
  public:

    // O(N^2) time complexity overall.
    static int brute_force(std::string s);
    static int count_substrings(std::string s);

    // O(N) time complexity.
    static bool is_palindrome(const std::string& s);
};

//------------------------------------------------------------------------------
/// 739. Daily Temperatures
/// Key themes: Traversing from the right. Stack. Check future values against
/// elements identified by index or "key" in stack.
//------------------------------------------------------------------------------
class DailyTemperatures
{
  public:

    static std::vector<int> brute_force(std::vector<int>& temperatures);

    //--------------------------------------------------------------------------
    /// Consider using a stack.
    /// Order preservation: problem requires to find the next occurrence of a
    /// specific condition (a higher temperature). Stacks are excellent for
    /// processing elements in reverse order (LIFO).
    //--------------------------------------------------------------------------
    static std::vector<int> daily_temperatures(std::vector<int>& temperatures);
};

//------------------------------------------------------------------------------
/// \name 2944. Minimum Number of Coins for Fruits
/// \url https://leetcode.com/problems/minimum-number-of-coins-for-fruits/
/// Dynamic programming.
/// \return minimum number of coins needed to acquire all the fruits.
/// Given a 1-indexed array prices, where prices[i] denotes number of coins
/// needed to purchase ith fruit.
/// If you purchase ith fruit at prices[i] coins, you can get next i fruits for
/// free.
/// Constraints:
/// 1 <= prices.length <= 1000
/// 1 <= prices[i] <= 10^5.
//------------------------------------------------------------------------------
class MinimumNumberOfCoinsForFruits
{
  public:

    // O(N^2) time complexity (iterate through all N fruits and for each fruit,
    // iterate through next i fruits).
    // O(N) space complexity to hold minimum_cost for each fruit of N fruits.
    static int minimum_coins(std::vector<int>& prices);
};

} // namespace LeetCode
} // namespace Algorithms

#endif // ALGORITHMS_LEETCODE_MEDIUM_PROBLEMS_H