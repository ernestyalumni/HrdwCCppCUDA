#ifndef ALGORITHMS_LEETCODE_MEDIUM_PROBLEMS_H
#define ALGORITHMS_LEETCODE_MEDIUM_PROBLEMS_H

#include "DataStructures/BinaryTrees.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
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
/// https://leetcode.com/problems/3sum/description/
/// 15. 3Sum
/// Given an integer array nums, return all the triplets
/// [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and
/// nums[i] + nums[j] + nums[k] == 0.
///
/// Notice that the solution set must not contain duplicate triplets.
///
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
/// 49. Group Anagrams
/// https://leetcode.com/problems/group-anagrams/description/
/// https://neetcode.io/problems/anagram-groups
//------------------------------------------------------------------------------
class GroupAnagrams
{
  public:

    // TODO: doesn't work, not sure why? :(
    /*
    static std::vector<std::vector<std::string>> group_anagrams(
      std::vector<std::string>& strs);
    */

    static std::vector<std::vector<std::string>> group_anagrams_by_sorting(
      std::vector<std::string>& strs);

    //--------------------------------------------------------------------------
    /// Distinguish between (types of) anagrams by counting the frequency (
    /// number of appearances?) of a letter for each anagram.
    ///
    /// We were given "strs[i] consists of lowercase English letters", so number
    /// of possible letters is bounded.
    //--------------------------------------------------------------------------
    static std::vector<std::vector<std::string>> group_anagrams_by_frequency(
      std::vector<std::string>& strs);

    static bool is_anagram_neet_code(
      const std::string& s,
      const std::string& t);

    static bool is_anagram_open_ai(const std::string& s, const std::string& t);

    // https://en.cppreference.com/w/cpp/utility/hash/operator()
    // std::hash<Key>::operator()
    // Takes a single argument key of type Key
    // returns value of type std::size_t that represents the hash value of key.
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
/// 57. Insert Interval
/// Given array of non-overlapping intervals, and intervals is sorted in
/// ascending order by start_i. You're also given an interval newInterval.
///
/// Insert newInterval into intervals such that intervals is still sorted in
/// ascending order by start and intervals still doesn't have any overlapping
/// intervals (merge overlapping intervals if necessary).
//------------------------------------------------------------------------------
class InsertInterval
{
  public:

    static std::size_t binary_search(
      std::vector<std::vector<int>>& intervals,
      std::vector<int>& new_interval);

    /*
    static std::vector<std::vector<int>> insert_with_binary_search(
      std::vector<std::vector<int>>& intervals,
      std::vector<int>& new_interval);
    */

    static std::vector<std::vector<int>> insert(
      std::vector<std::vector<int>>& intervals,
      std::vector<int>& new_interval);
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
/// 98. Validate Binary Search Tree
/// Key ideas: Recursion on left subtree, right subtree. Keep track of 2 int's
/// for lower bound and upper bound so all values in subtree is either smaller
/// (on left subtree) or greater (on right subtree) of node.
//------------------------------------------------------------------------------
class ValidateBinarySearchTree
{
  public:

    using TreeNode = DataStructures::BinaryTrees::TreeNode;

    static bool is_valid_BST(TreeNode* root);

    // Use two more TreeNode pointers in the arguments to track lower and upper
    // bounds.
    static bool is_valid_BST_track_parent_pointer(TreeNode* root);
};

//------------------------------------------------------------------------------
/// https://leetcode.com/problems/binary-tree-level-order-traversal/
/// 102. Binary Tree Level Order Traversal
///
/// Given the root of a binary tree, return the level order traversal of its
/// nodes' values. (i.e., from left to right, level by level).
///
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
/// 128. Longest Consecutive Sequence
/// Given an unsorted array of integers nums, return the length of the longest
/// consecutive elements sequence.
///
/// You must write an algorithm that runs in O(n) time.
/// https://leetcode.com/problems/longest-consecutive-sequence/description/
/// https://neetcode.io/problems/longest-consecutive-sequence
/// https://youtu.be/P6RZZMu_maU?si=lxavcMLBZPAunM7M
/// Key idea is that each sequence is separated by at least one element, i.e.
/// each start element has no left neighbor; likewise each end element has no
/// right neighbor.
//------------------------------------------------------------------------------

class LongestConsecutiveSequence
{
  public:

    static int longest_consecutive_with_set(std::vector<int>& nums);
};

//------------------------------------------------------------------------------
/// 137. Single Number II
/// https://leetcode.com/problems/single-number-ii/description/
/// Given an integer array nums where every element appears 3 times except for
/// one, which appears exactly once. Find single element and return it.
//------------------------------------------------------------------------------
class SingleNumberII
{
  public:

    static int single_number_count_per_bit(std::vector<int>& nums);
    static int single_number_track_seen(std::vector<int>& nums);
};

//------------------------------------------------------------------------------
/// 146. LRU Cache
/// Design a data structure that follows the constraints of a Least Recently
/// Used (LRU) cache.
///
/// Implement the LRUCache class:
///
/// * LRUCache(int capacity) Initialize the LRU cache with positive size
/// capacity.
/// * int get(int key) Return the value of the key if the key exists, otherwise
/// return -1.
/// void put(int key, int value) Update the value of the key if the key exists.
/// Otherwise, add the key-value pair to the cache. If the number of keys
/// exceeds the capacity from this operation, evict the least recently used key.
/// The functions get and put must each run in O(1) average time complexity.
///
/// Constraints:
///
/// * 1 <= capacity <= 3000
/// * 0 <= key <= 104
/// * 0 <= value <= 105
/// * At most 2 * 105 calls will be made to get and put.
//------------------------------------------------------------------------------

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
/// 155. Min Stack
/// https://leetcode.com/problems/min-stack/
/// Design a stack that supports push, pop, top, and retrieving the minimum
/// element in constant time.
///
/// Implement the MinStack class:
///
///    MinStack() initializes the stack object.
///    void push(int val) pushes the element val onto the stack.
///    void pop() removes the element on the top of the stack.
///    int top() gets the top element of the stack.
///    int getMin() retrieves the minimum element in the stack.
///
/// You must implement a solution with O(1) time complexity for each function.
//------------------------------------------------------------------------------
class MinStack
{
  public:

    MinStack();

    void push(int val);
    
    void pop();
    
    int top();
    
    int getMin();

  protected:

    static constexpr int MAXIMUM_CALLS_ {30'000};

    bool is_empty();

  private:

    std::vector<int> array_;

    int top_index_;

    std::optional<std::pair<int, int>> minimum_value_and_count_;
};

//------------------------------------------------------------------------------
/// 167. Two Sum II - Input Array Is Sorted
/// The tests are generated such that there is exactly one solution. You may not
/// use the same element twice.
/// https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/description/
///
/// Return the indices of the two numbers, index1 and index2, added by one as an
/// integer array [index1, index2] of length 2.
///
/// numbers is sorted in non-decreasing order.
//------------------------------------------------------------------------------
class TwoSumII
{
  public:

    static std::vector<int> two_sum(std::vector<int>& numbers, int target);
};

//------------------------------------------------------------------------------
/// 187. Repeated DNA Sequences
/// https://leetcode.com/problems/repeated-dna-sequences/description/
/// The DNA sequence is composed of a series of nucleotides abbreviated as 'A',
/// 'C', 'G', and 'T'.
///
/// For example, "ACGAATTCCG" is a DNA sequence.
///
/// When studying DNA, it is useful to identify repeated sequences within the
/// DNA.
///
/// Given a string s that represents a DNA sequence, return all the
/// 10-letter-long sequences (substrings) that occur more than once in a DNA
/// molecule. You may return the answer in any order.
///
/// Constraints:
///
/// 1 <= s.length <= 105
/// s[i] is either 'A', 'C', 'G', or 'T'.
//------------------------------------------------------------------------------
class RepeatedDNASequences
{
  public:

    //--------------------------------------------------------------------------
    /// Leetcode: Runtime
    /// 35ms
    /// Beats82.71%
    //--------------------------------------------------------------------------
    static std::vector<std::string> find_repeated_dna_sequences(std::string s);
};

//------------------------------------------------------------------------------
/// 200. Number of Islands
/// Key idea: Graph. Map each matrix element / matrix cell to a vertex on the
/// graph. Map the allowed (watch for boundaries) directions, up, down, left,
/// right, i.e. horizontal, vertical adjacent, to edges between vertices.
/// An island is surrounded by water, so if we hit water, we can end looking
/// from there.
///
/// Key insight is to reuse the grid and that it's ok to mark a visited element
/// with a '0' for seen.
//------------------------------------------------------------------------------
class NumberOfIslands
{
  public:

    static int number_of_islands_with_depth_first_search(
      std::vector<std::vector<char>>& grid);

    static int number_of_islands_with_breadth_first_search(
      std::vector<std::vector<char>>& grid);
};

//------------------------------------------------------------------------------
/// 201. Bitwise AND of Numbers Range
/// https://leetcode.com/problems/bitwise-and-of-numbers-range/description/
/// Given 2 integers left and right that represent the range [left, right],
/// return bitwise AND of all numbers in this range, inclusive.
//------------------------------------------------------------------------------
class BitwiseANDOfNumbersRange
{
  public:

    static int naive_loop(int left, int right);

    static int range_bitwise_and(int left, int right);
    static int common_mask(int left, int right);
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
/// 230. Kth Smallest Element in a BST
/// Key idea: by the property of the BST, by induction, we expect that the
/// smallest element is all the way to the left.
//------------------------------------------------------------------------------
class KthSmallestElementInABST
{
  public:

    using TreeNode = DataStructures::BinaryTrees::TreeNode;

    //--------------------------------------------------------------------------
    /// Key idea: Use a current TreeNode* pointer to track the smallest
    /// elements.
    /// From each node, check the most left path of only left children for the
    /// smallest. Use a stack to push into the stack all the left children of
    /// this path.
    //--------------------------------------------------------------------------
    static int kth_smallest_iterative(TreeNode* root, int k);

    static int kth_smallest_recursive(TreeNode* root, int k);
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
/// 260. Single Number III
/// https://leetcode.com/problems/single-number-iii/description/
/// Given an integer array nums, in which exactly 2 elements appear only once
/// and all other elements appear exactly twice. Find the 2 elements that appear
/// only once.
//------------------------------------------------------------------------------
class SingleNumberIII
{
  public:

    std::vector<int> single_number(std::vector<int>& nums);
};

//------------------------------------------------------------------------------
/// 271. String Encode and Decode
/// https://neetcode.io/problems/string-encode-and-decode
/// https://leetcode.com/problems/encode-and-decode-strings/description/
/// Design an algorithm to encode a list of strings to a single string. The
/// encoded string is then decoded back to the original list of strings.
/// Constraints:
/// 0 <= strs.length < 100
/// 0 <= strs[i].length < 200
/// strs[i] contains only UTF-8 characters.
//------------------------------------------------------------------------------
class StringEncodeAndDecode
{
  public:

    static std::string encode(std::vector<std::string>& strs);

    static std::vector<std::string> decode(std::string s);

    // https://youtu.be/B1k_sxOSgv8

    static std::string encode_with_prefix_neet(std::vector<std::string>& strs);

    static std::vector<std::string> decode_with_prefix_neet(std::string s);
};

//------------------------------------------------------------------------------
/// 289. Game of Life
/// https://leetcode.com/problems/game-of-life/description/
/// According to Wikipedia's article: "The Game of Life, also known simply as
/// Life, is a cellular automaton devised by the British mathematician John
/// Horton Conway in 1970."
///
/// The board is made up of an m x n grid of cells, where each cell has an
/// initial state: live (represented by a 1) or dead (represented by a 0). Each
/// cell interacts with its eight neighbors (horizontal, vertical, diagonal)
/// using the following four rules (taken from the above Wikipedia article):
///
/// Any live cell with fewer than two live neighbors dies as if caused by
/// under-population.
/// Any live cell with two or three live neighbors lives on to the next
/// generation.
/// Any live cell with more than three live neighbors dies, as if by
/// over-population.
/// Any dead cell with exactly three live neighbors becomes a live cell, as if
/// by reproduction.
/// The next state of the board is determined by applying the above rules
/// simultaneously to every cell in the current state of the m x n grid board.
/// In this process, births and deaths occur simultaneously.
///
/// Given the current state of the board, update the board to reflect its next
/// state.
///
/// Note that you do not need to return anything.
//------------------------------------------------------------------------------

class GameOfLife
{
  public:

    static void game_of_life(std::vector<std::vector<int>>& board);
};

//------------------------------------------------------------------------------
/// \name 347. Top K Frequent Elements
/// Given an integer array nums and an integer k, return the k most frequent
/// elements. You may return the answer in any order.
///
/// 1 <= nums.length <= 10^5
/// EY: But with if nums.length is 10^9 (billions)? This is the case of for
/// example a billion+ Youtube videos and you want to obtain the K most popular
/// Youtube videos.
//------------------------------------------------------------------------------
class TopKFrequentElements
{
  public:

    //std::vector<int> top_k_frequent(std::vector<int>& nums, int k);

    static std::vector<int> brute_force(std::vector<int>& nums, int k);

    // https://youtu.be/YPTqKIgVk-k?si=KbHT9oTwGare3dkl
    static std::vector<int> bucket_sort(std::vector<int>& nums, int k);

    static std::vector<int> with_max_heap(std::vector<int>& nums, int k);
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
/// 371. Sum of Two Integers
/// https://leetcode.com/problems/sum-of-two-integers/description/
//------------------------------------------------------------------------------
class SumOfTwoIntegers
{
  public:

    static int get_sum(int a, int b);
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
/// https://leetcode.com/problems/longest-repeating-character-replacement/description/
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
/// \name 542. 01 Matrix
/// \url https://leetcode.com/problems/01-matrix/
//------------------------------------------------------------------------------
class Update01Matrix
{
  public:

    static std::vector<std::vector<int>> update_matrix(
      std::vector<std::vector<int>>& mat);
};

//------------------------------------------------------------------------------
/// \name 547. Number of Provinces
/// \url https://leetcode.com/problems/number-of-provinces/
//------------------------------------------------------------------------------
int find_number_of_provinces(std::vector<std::vector<int>>& is_connected);

//------------------------------------------------------------------------------
/// \name 567. Permutation in String
/// Given two strings s1 and s2, return true if s2 contains a permutation of s1,
/// or false otherwise.
///
/// In other words, return true if one of s1's permutations is the substring of
/// s2.
/// https://leetcode.com/problems/permutation-in-string/description/
///
/// s1 and s2 consist of lowercase English letters.
//------------------------------------------------------------------------------
class PermutationInString
{
  public:

    //--------------------------------------------------------------------------
    /// Idea: Keep 2 arrays for 0(1) look up of lowercase letter to count for
    /// each strings s1, s2 and compare them after the window length is of s1.
    /// https://youtu.be/UbyhOgBN834?si=_LVBkImAFusoN8DK
    //--------------------------------------------------------------------------
    static bool check_inclusion(std::string s1, std::string s2);
};

//------------------------------------------------------------------------------
/// \name 621. Task Scheduler
/// \url https://leetcode.com/problems/task-scheduler/description/
/// You are given an array of CPU tasks, each labeled with a letter from A to Z,
/// and a number n. Each CPU interval can be idle or allow the completion of one
/// task. Tasks can be completed in any order, but there's a constraint: there
/// has to be a gap of at least n intervals between two tasks with the same
/// label.
///
/// Return the minimum number of CPU intervals required to complete all tasks.
///
/// Constraints:
///
/// 1 <= tasks.length <= 104
/// tasks[i] is an uppercase English letter.
/// 0 <= n <= 100
//------------------------------------------------------------------------------
class TaskScheduler
{
  public:

    //--------------------------------------------------------------------------
    /// Observe: Because the tasks are guaranteed to be uppercase English
    /// letters, use an array of fixed size 26.
    /// For every cycle, find the most frequent letter that can be placed in
    /// this cycle. After placing, decrease the frequency of that letter by one.
    //--------------------------------------------------------------------------
    static int with_min_heap(std::vector<char>& tasks, int n);

    static int least_interval_by_math(std::vector<char>& tasks, int n);
};

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
/// 692. Top K Frequent Words
/// https://leetcode.com/problems/top-k-frequent-words/description/
/// Given an array of strings words and an integer k, return the k most frequent
/// strings.
///
/// Return the answer sorted by the frequency from highest to lowest. Sort the
/// words with the same frequency by their lexicographical order.
///
/// Constraints:
///
/// 1 <= words.length <= 500
/// 1 <= words[i].length <= 10
/// words[i] consists of lowercase English letters.
/// k is in the range [1, The number of unique words[i]]
///
/// Follow-up: Could you solve it in O(n log(k)) time and O(n) extra space?
//------------------------------------------------------------------------------
class TopKFrequentWords
{
  public:

    //--------------------------------------------------------------------------
    /// O(N log N) time.
    /// Leetcode: Runtime
    /// 4ms
    /// Beats 44.72%
    /// Memory
    /// 17.27 MB
    /// Beats 77.01%
    //--------------------------------------------------------------------------
    static std::vector<std::string> brute_force(
      std::vector<std::string>& words,
      int k);  

    static std::vector<std::string> top_k_frequent(
      std::vector<std::string>& words,
      int k);
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
/// 1297. Maximum Number of Occurrences of a Substring
///
/// Given a string s, return the maximum number of occurrences of any substring
/// under the following rules:
///
/// The number of unique characters in the substring must be less than or equal
/// to maxLetters.
/// The substring size must be between minSize and maxSize inclusive.
///
/// Constraints:
///
/// 1 <= s.length <= 105
/// 1 <= maxLetters <= 26
/// 1 <= minSize <= maxSize <= min(26, s.length)
/// s consists of only lowercase English letters.
///
/// Key observations:
/// Observe that the substring with the most occurrences must always be a
/// a substring of minimal size.
/// Proof by contradiction: if a substring is of size min size + 1 and has
/// maximum occurrence, each of those occurrences contains a substring of size
/// min size and would occur at least as much as substing of size min size + 1.
//------------------------------------------------------------------------------
class MaximumNumberOfOccurrencesOfASubstring
{
  public:

    static int max_freq(
      std::string s,
      int max_letters,
      int min_size,
      int max_size);

    //--------------------------------------------------------------------------
    /// Leetcode: Runtime
    /// 31ms
    /// Beats83.88%
    //--------------------------------------------------------------------------
    static int max_freq_with_bitfield(
      std::string s,
      int max_letters,
      int min_size,
      int max_size);

    //--------------------------------------------------------------------------
    /// Leetcode: Runtime
    /// 23ms
    /// Beats94.89%
    //--------------------------------------------------------------------------
    static int max_freq_with_sliding_window(
      std::string s,
      int max_letters,
      int min_size,
      int max_size);
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