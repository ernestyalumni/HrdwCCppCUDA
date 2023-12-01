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