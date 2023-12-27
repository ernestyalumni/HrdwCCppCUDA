#include "EasyProblems.h"

#include <algorithm> // std::max
#include <limits.h> // INT_MIN
#include <map>
#include <unordered_set>
#include <vector>

using std::map;
using std::max;
using std::unordered_set;
using std::vector;

namespace Algorithms
{
namespace LeetCode
{

//------------------------------------------------------------------------------
/// 1. Two Sum
//------------------------------------------------------------------------------

vector<int> TwoSum::brute_force(vector<int>& nums, int target)
{
  const int N {static_cast<int>(nums.size())};

  // O(N^2) time complexity.
  // Given N number of integers, for all pairs of i, j \in 0 .. N - 1, i 1= j
  // find i, j such that nums[i] + nums[j] = target.
  for (int i {0}; i < N - 1; ++i)
  {
    for (int j {i + 1}; j < N; ++j)
    {
      if (nums[i] + nums[j] == target)
      {
        return vector<int>{i, j};
      }
    }
  }

  return vector<int>{};
}

vector<int> TwoSum::two_sum(vector<int>& nums, int target)
{
  const int N {static_cast<int>(nums.size())};

  // Use another data structure to store progress as we traverse the array nums.
  std::map<int, int> value_and_indices {};

  for (int i {0}; i < N; ++i)
  {
    const int complement {target - nums[i]};

    if (value_and_indices.count(complement) > 0)
    {
      return vector<int>{i, value_and_indices[complement]};
    }
    else
    {
      value_and_indices.emplace(nums[i], i);
    }
  }

  return vector<int>{};
}

//------------------------------------------------------------------------------
/// 88. Merge Sorted Array
//------------------------------------------------------------------------------

void MergeSortedArray::merge(
  vector<int>& nums1,
  int m,
  vector<int>& nums2,
  int n)
{
  if (n == 0)
  {
    return;
  }

  if (m == 0)
  {
    nums1 = nums2;
  }

  // The key insight is to start from the end and we know from the end and
  // decrementing, we obtain the largest, and non-increasing.

  int current_index_1 {m - 1};
  int current_index_2 {n - 1};
  int tail {m + n - 1};

  while (tail >= 0)
  {
    if (current_index_1 >= 0 && current_index_2 >= 0)
    {
      if (nums1[current_index_1] > nums2[current_index_2])
      {
        nums1[tail] = nums1[current_index_1];
        --current_index_1;
      }
      else
      {
        nums1[tail] = nums2[current_index_2];
        --current_index_2;
      }

      --tail;
    }
    else if (current_index_2 >= 0)
    {
      while (current_index_2 >= 0)
      {
        nums1[tail] = nums2[current_index_2];
        --current_index_2;
        --tail;
      }
    }
    // Otherwise nums1 is already in non-decreasing order.
    else
    {
      --tail;
    }
  }
}

//------------------------------------------------------------------------------
/// 121. Best Time to Buy and Sell Stock
//------------------------------------------------------------------------------

int BestTimeToBuyAndSellStock::max_profit(vector<int>& prices)
{
  const int N {static_cast<int>(prices.size())};
  int minimum_price {prices[0]};
  int profit {0};

  for (int i {0}; i < N; ++i)
  {
    const int current_profit {prices[i] - minimum_price};

    if (current_profit > profit)
    {
      profit = current_profit;
    }

    if (prices[i] < minimum_price)
    {
      minimum_price = prices[i];
    }
  }

  return profit;
}

//------------------------------------------------------------------------------
/// 217. Contains Duplicate
//------------------------------------------------------------------------------

bool ContainsDuplicate::contains_duplicate(vector<int>& nums)
{
  unordered_set<int> seen_numbers {};

  for (const auto num : nums)
  {
    // O(1) time complexity, amoritized.
    if (seen_numbers.count(num) == 0)
    {
      seen_numbers.emplace(num);
    }
    else
    {
      return true;
    }
  }

  return false;
}

//------------------------------------------------------------------------------
/// 1646. Get Maximum in Generated Array.
//------------------------------------------------------------------------------

int GetMaximumInGeneratedArray::get_maximum_generated(int n)
{
  if (n == 0 || n == 1)
  {
    return n;
  }

  // Use -1 as a value to show that there wasn't a value before.
  std::vector<int> values (n + 1, -1);

  values[0] = 0;
  values[1] = 1;
  int maximum {1};

  // O(N) time complexity.
  for (int i {2}; i < n + 1; ++i)
  {
    if (values[i] == -1)
    {
      // i is even,
      if (i % 2 == 0)
      {
        values[i] = values[i / 2];
      }
      // i is odd
      else
      {
        values[i] = values[i / 2] + values[i / 2 + 1];
      }
    }

    if (values[i] > maximum)
    {
      maximum = values[i];
    }
  }

  return maximum;
}

} // namespace LeetCode
} // namespace Algorithms
