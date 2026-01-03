#ifndef ALGORITHMS_LEETCODE_EASY_PROBLEMS_H
#define ALGORITHMS_LEETCODE_EASY_PROBLEMS_H

#include "DataStructures/BinaryTrees.h"

#include <string>
#include <vector>

namespace Algorithms
{
namespace LeetCode
{

//------------------------------------------------------------------------------
/// 1. Two Sum
/// https://leetcode.com/problems/two-sum/  
/// Constraints 2 <= nums.length <= 10^4
/// -10^9 <= nums[i], target <= 10^9
//------------------------------------------------------------------------------
class TwoSum
{
  public:

    //--------------------------------------------------------------------------
    /// O(N^2) time complexity, 2 nested for loops to get each value.
    //--------------------------------------------------------------------------
    static std::vector<int> brute_force(std::vector<int>& nums, int target);

    //--------------------------------------------------------------------------
    /// O(N) time, O(N) space.
    //--------------------------------------------------------------------------
    static std::vector<int> two_sum(std::vector<int>& nums, int target);
};

//------------------------------------------------------------------------------
/// 20. Valid Parentheses
/// https://leetcode.com/problems/valid-parentheses/
/// s consists of parentheses only '()[]{}'.
//------------------------------------------------------------------------------
class ValidParentheses
{
  public:

    static bool is_valid(std::string s);
};

//------------------------------------------------------------------------------
/// 88. Merge Sorted Array
/// Given 2 integer arrrays sorted in non-decreasing order.
//------------------------------------------------------------------------------
class MergeSortedArray
{
  public:

    static void merge(
      std::vector<int>& nums1,
      int m,
      std::vector<int>& nums2,
      int n);
};

//------------------------------------------------------------------------------
/// 100. Same Tree
//------------------------------------------------------------------------------
class SameTree
{
  public:

    using TreeNode = DataStructures::BinaryTrees::TreeNode;

    static bool is_same_tree(TreeNode* p, TreeNode* q);
};

//------------------------------------------------------------------------------
/// 104. Maximum Depth of Binary Tree
/// A binary tree's maximum depth is the number of nodes along the longest path
/// from the root node down to the farthest leaf node.
//------------------------------------------------------------------------------
class MaximumDepthOfBinaryTree
{
  public:

    using TreeNode = DataStructures::BinaryTrees::TreeNode;

    //--------------------------------------------------------------------------
    /// Key idea: solve the easiest base case. Let recursion solve the left case
    /// and the right child case.
    //--------------------------------------------------------------------------
    static int max_depth_recursive(TreeNode* root);

    //--------------------------------------------------------------------------
    /// Key idea: Use a queue.
    //--------------------------------------------------------------------------
    static int max_depth_iterative(TreeNode* root);
};

//------------------------------------------------------------------------------
/// 121. Best Time to Buy and Sell Stock
/// Constraints:
/// 1 <= prices.length <= 10^5
/// 0 <= prices[i] <= 10^4
/// Key idea: at each step update profit for maximum profit and minimum price in
/// that order.
//------------------------------------------------------------------------------
class BestTimeToBuyAndSellStock
{
  public:

    static int max_profit(std::vector<int>& prices);
};

//------------------------------------------------------------------------------
/// 125. Valid Palindrome
/// Linearly 
//------------------------------------------------------------------------------
class ValidPalindrome
{
  public:

    static bool is_palindrome(std::string s);
};

//------------------------------------------------------------------------------
/// 136. Single Number
/// Given a non-empty array of integers (nums), every element appears twice
/// except for one. Find that single one.
/// https://leetcode.com/problems/single-number/description/
//------------------------------------------------------------------------------
class SingleNumber
{
  public:

    static int single_number_with_set(std::vector<int>& nums);
    //--------------------------------------------------------------------------
    /// Key insight: XOR is a group operator for the group of 32-bit vectors,
    /// such that
    /// 1. identity exists, 0 (x ^ 0 = x)
    /// 2. associativity
    /// 3. commutativity
    /// 4. self-inverse, for each x, inverse is itself (x ^ x = 0)
    /// So for this problem,
    /// a ^ b ^ ... ^ b ^ a ... ^ s = (a ^ a) ^ (b ^ b) ^ ... ^ s = 0 ^ 0 ^ s =
    /// s.
    //--------------------------------------------------------------------------
    static int single_number_xor(std::vector<int>& nums);
};


//------------------------------------------------------------------------------
/// 169. Majority Element
/// Array - Linear traversal.
/// 1. Use std::unordered_map for O(1) amortized lookup and insertion and count
/// frequencies.
/// 2. Boyer-Moore Voting Algorithm - you are guaranteed a majority element that
/// appears more than floor(n/2) times in an array. Let all other elements be a
/// non-majority element, x[j]. Let a majority element be x[i].
/// For each x[j], there exists x[i]. Pair each x[j] with an x[i]. Because there
/// is more majority elements than any non-majority element, there exists at
/// least one x[i] that won't be paired up with an x[j].
/// i.e. since x[i] appears more than floor(n/2) times, the candidate will
/// eventually revert to x[i] and accumulate more counts than any other element,
/// regardless of order.
//------------------------------------------------------------------------------
class MajorityElement
{
  public:

    static int majority_element_with_map(std::vector<int>& nums);

    static int majority_element_with_voting(std::vector<int>& nums);
};

//------------------------------------------------------------------------------
/// 190. Reverse Bits
/// Reverse bits of a given 32 bits unsigned integer.
/// https://leetcode.com/problems/reverse-bits/description/
//------------------------------------------------------------------------------
class ReverseBits
{
  public:

    static int reverse_bits_loop_through(int n);
    static int reverse_bits_get_and_shift_lsb(int n);
    static int reverse_bits_swap_halves(int n);
};

//------------------------------------------------------------------------------
/// 191. Number of 1 Bits
/// Returns number of set bits in its binary representation (also known as
/// Hamming weight)
/// http://en.wikipedia.org/wiki/Hamming_weight
//------------------------------------------------------------------------------
class NumberOf1Bits
{
  public:

    static int hamming_weight_loop_all_bits(int n);
    static int hamming_weight_kernighan_trick(int n);
};

//------------------------------------------------------------------------------
/// 217. Contains Duplicate
/// Given an integer array nums, return true if any value appears at least twice
/// in the array, and return false if every element is distinct.
///
/// Key ideas: unordered_set<int> to track duplicate values. 
/// Or you can sort. Then 1 pass is enough. Because the duplicate would be next
/// to the digit.
//------------------------------------------------------------------------------
class ContainsDuplicate
{
  public:

    //--------------------------------------------------------------------------
    /// O(N) time, O(N) space (store all distinct values)
    //--------------------------------------------------------------------------
    static bool contains_duplicate(std::vector<int>& nums);

    // O(nlogn) (with sorting), O(1) space.
    static bool sort_first(std::vector<int>& nums);
};

//------------------------------------------------------------------------------
/// 226. Invert Binary Tree
//------------------------------------------------------------------------------
class InvertBinaryTree
{
  public:

    using TreeNode = DataStructures::BinaryTrees::TreeNode;

    static TreeNode* invert_tree_recursive(TreeNode* root);

    static TreeNode* invert_tree_iterative(TreeNode* root);
};

//------------------------------------------------------------------------------
/// 231. Power of Two
//------------------------------------------------------------------------------
class PowerOfTwo
{
  public:
    static bool is_power_of_two_and(int n);
};

//------------------------------------------------------------------------------
/// 242. Valid Anagram
//------------------------------------------------------------------------------
class ValidAnagram
{
  public:

    static bool is_anagram(std::string s, std::string t);
};

//------------------------------------------------------------------------------
/// 268. Missing Number
/// https://leetcode.com/problems/missing-number/description/
//------------------------------------------------------------------------------
class MissingNumber
{
  public:

    static int missing_number_xor(std::vector<int>& nums);
    static int missing_number_sum_formula(std::vector<int>& nums);
};

//------------------------------------------------------------------------------
/// 338. Counting Bits
/// https://leetcode.com/problems/counting-bits/description/
/// Given an integer n, return an array ans of length n + 1 such that for each
/// i (0 <= i <= n), ans[[i] is the number of 1's in the binary representation
/// of i.
//------------------------------------------------------------------------------
class CountingBits
{
  public:

    static std::vector<int> count_bits_memoization(int n);
};

//------------------------------------------------------------------------------
/// 405. Convert a Number to Hexadecimal
/// https://leetcode.com/problems/convert-a-number-to-hexadecimal/description/
/// Given a 32-bit integer num, return a string representing its hexadecimal
/// representation. For negative integers, two's complement method is used.
//------------------------------------------------------------------------------
class ConvertToHexadecimal
{
  public:

    static std::string to_hex(int num);
};

//------------------------------------------------------------------------------
/// 461. Hamming Distance
/// https://leetcode.com/problems/hamming-distance/description/
/// The Hamming distance between 2 integers is the number of positions at which
/// the corresponding bits are different.
//------------------------------------------------------------------------------
class HammingDistance
{
  public:

    static int hamming_distance(int x, int y);
};

//------------------------------------------------------------------------------
/// 704. Binary Search
//------------------------------------------------------------------------------
class BinarySearch
{
  public:

    static int search(std::vector<int>& nums, int target);
};

//------------------------------------------------------------------------------
/// 733. Flood Fill
//------------------------------------------------------------------------------
class FloodFill
{
  public:

    static std::vector<std::vector<int>> flood_fill(
      std::vector<std::vector<int>>& image, int sr, int sc, int color);

    static std::vector<std::vector<int>> flood_fill_with_queue(
      std::vector<std::vector<int>>& image, int sr, int sc, int color);
};

//------------------------------------------------------------------------------
/// 1646. Get Maximum in Generated Array.
/// Given integer n, A 0-indexed integer array nums of length n + 1 is generated
/// in the following way:
/// nums[0] = 0
/// nums[1] = 1
/// nums[2 * i] = nums[i] when 2 <= 2 * i <= n
/// nums[2 * i + 1] = nums[i] + nums[i + 1 when 2 <= 2 * i + 1 <= n]
/// Return maximum integer in array nums.
///
/// EY: the rules seem to be monotonically increasing with i, and so we expect
/// the maximum to be "near" the end of that array.
///
/// Assume that 0 <= n <= 100
//------------------------------------------------------------------------------
class GetMaximumInGeneratedArray
{
  public:

    //--------------------------------------------------------------------------
    /// O(N) time complexity.
    //--------------------------------------------------------------------------
    static int get_maximum_generated(int n);
};

} // namespace LeetCode
} // namespace Algorithms

#endif // ALGORITHMS_LEETCODE_EASY_PROBLEMS_H