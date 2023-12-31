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

    static std::vector<int> brute_force(std::vector<int>& nums, int target);

    static std::vector<int> two_sum(std::vector<int>& nums, int target);
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
/// 217. Contains Duplicate
/// Key ideas: unordered_set<int> to track duplicate values. 
//------------------------------------------------------------------------------
class ContainsDuplicate
{
  public:

    static bool contains_duplicate(std::vector<int>& nums);
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
/// 242. Valid Anagram
//------------------------------------------------------------------------------
class ValidAnagram
{
  public:

    static bool is_anagram(std::string s, std::string t);
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