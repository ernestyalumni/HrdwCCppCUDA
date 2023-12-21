#include "EasyProblems.h"

#include <map>
#include <vector>

using std::map;
using std::vector;

namespace Algorithms
{
namespace LeetCode
{

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
