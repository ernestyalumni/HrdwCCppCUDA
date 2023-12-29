#ifndef ALGORITHMS_LEETCODE_HARD_PROBLEMS_H
#define ALGORITHMS_LEETCODE_HARD_PROBLEMS_H

#include <string>
#include <vector>

namespace Algorithms
{
namespace LeetCode
{

//------------------------------------------------------------------------------
/// 41. First Missing Positive
/// Constraints
/// 1 <= nums.length <= 10^5
/// -2^31 <= nums[i] <= 2^31 - 1
//------------------------------------------------------------------------------
class FirstMissingPositive
{
  public:

    static int first_missing_positive(std::vector<int>& nums);
};

//------------------------------------------------------------------------------
/// \name 76. Minimum Window Substring
/// Return minimum window substring of s such that every character in t
/// (including duplicates) is included in window. If tehre's no such substring,
/// return empty string "".
/// Constraints
/// s and t consist of uppercase and lowercase English letters.
//------------------------------------------------------------------------------
class MinimumWindowSubstring
{
  public:

    //--------------------------------------------------------------------------
    /// \details Sliding Window technique, use hash tables via
    /// std::unordered_map. O(K) space complexity for K = number of unique
    /// characters in string t.
    //--------------------------------------------------------------------------
    static std::string minimum_window(std::string s, std::string t);
};

//------------------------------------------------------------------------------
/// 239. Sliding Window Maximum
/// Constraints:
/// 1 <= k <= nums.length
/// Key ideas: Use a deque and track the sliding window maximum as top of deque.
/// Keeping adding to deque until it fills up the k sized sliding window.
//------------------------------------------------------------------------------
class SlidingWindowMaximum
{
  public:

    static std::vector<int> max_sliding_window(std::vector<int>& nums, int k);
};

//------------------------------------------------------------------------------
/// 1293. Shortest Path in a Grid with Obstacles Elimination
//------------------------------------------------------------------------------
class ShortestPathInGridWithObstacles
{
  public:

    static int shortest_path(std::vector<std::vector<int>>& grid, int k);
};

//------------------------------------------------------------------------------
/// 1547. Minimum Cost to Cut a Stick.
/// Given a wooden stick of length n units, stick labelled from 0 to n.
/// Given integer array cuts in order, but can change order of cuts.
/// Cost of 1 cut is length of stick to be cut, total cost is sum of costs of
/// all cuts. When you cut a stick, i'll be split into 2 smaller sticks (the
/// sum of their lengths is length of stick before cut).
/// \return minimum total cost of the cuts.
/// Constraints:
/// 2 <= n <= 10^6
/// 1 <= cuts.length <= min(n-1, 100)
/// 1 <= cuts[i] <- n - 1
/// All integers in cuts array are distinct.
//------------------------------------------------------------------------------
class MinimumCostToCutStick
{
  public:

    static int minimum_cost_to_cut_stick(int n, std::vector<int>& cuts);
};

//------------------------------------------------------------------------------
/// 1944. Number of Visible People in a Queue
/// Key insight is to iterate *from the right* or "from the end" because "A
/// person can see another person to their right in the queue if everybody in
/// between is shorter than both of them. More formally, ith person can see the
/// jth person if i < j and min(heights[i], heights[j]) > max(heights[i+1],
/// heights[i+2], ..., heights[j-1])"
///
/// Keywords: Traversing from the right. Stack.
///
/// Another key insight is to use a stack such that Last In First Out applies.
//------------------------------------------------------------------------------
class NumberOfVisiblePeopleInAQueue
{
  public:

    static std::vector<int> can_see_persons_count(std::vector<int>& heights);
};

//------------------------------------------------------------------------------
/// 2585. Number of Ways to Earn Points.
/// Test has n types of questions. 
/// You're given integer target and 0-indexed 2D integer array types where
/// types[i] = [count, marks] indicates there are count questions of ith type,
/// and each is worth marks points.
/// Constraints:
/// 1 <= n <= 50 where n == types.length.
/// 1 <= count, marks <= 50
/// EY: Are the points or marks unique?
//------------------------------------------------------------------------------
class WaysToEarnPoints
{
  public:

    static int ways_to_reach_target(
      int target,
      std::vector<std::vector<int>>& types);
};

} // namespace LeetCode
} // namespace Algorithms

#endif // ALGORITHMS_LEETCODE_HARD_PROBLEMS_H