#ifndef ALGORITHMS_LEETCODE_EASY_PROBLEMS_H
#define ALGORITHMS_LEETCODE_EASY_PROBLEMS_H

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