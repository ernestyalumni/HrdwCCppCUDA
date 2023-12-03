#ifndef ALGORITHMS_LEETCODE_HARD_PROBLEMS_H
#define ALGORITHMS_LEETCODE_HARD_PROBLEMS_H

#include <vector>

namespace Algorithms
{
namespace LeetCode
{

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

} // namespace LeetCode
} // namespace Algorithms

#endif // ALGORITHMS_LEETCODE_HARD_PROBLEMS_H