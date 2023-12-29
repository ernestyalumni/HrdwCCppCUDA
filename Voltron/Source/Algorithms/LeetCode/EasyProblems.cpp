#include "EasyProblems.h"

#include <algorithm> // std::max
#include <limits.h> // INT_MIN
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

using std::map;
using std::max;
using std::string;
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
/// Key idea: at each step update profit for maximum profit and minimum price in
/// that order.
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
/// 125. Valid Palindrome
//------------------------------------------------------------------------------

bool ValidPalindrome::is_palindrome(string s)
{
  const int valid_min {static_cast<int>('a')};
  const int valid_max {static_cast<int>('z')};

  // Numbers are ok ("alphanumeric characters include letters and numbers.")
  const int valid_numbers_min {static_cast<int>('0')};
  const int valid_numbers_max {static_cast<int>('9')};

  const int valid_upper_case_min {static_cast<int>('A')};
  const int valid_upper_case_max {static_cast<int>('Z')};

  // O(|s|) space complexity.
  vector<char> stripped_s {};

  // O(|s|) time complexity.
  for (const char c : s)
  {
    const int c_value {static_cast<int>(c)};

    if ((c_value <= valid_max && c_value >= valid_min) ||
      (c_value <= valid_numbers_max && c_value >= valid_numbers_min))
    {
      stripped_s.emplace_back(c);
    }
    else if (c_value <= valid_upper_case_max && c_value >= valid_upper_case_min)
    {
      stripped_s.emplace_back(c - ('A' - 'a'));
    }
  }

  int l {0};
  int r {static_cast<int>(stripped_s.size()) - 1};
  while (l <= r)
  {
    if (stripped_s[l] != stripped_s[r])
    {
      return false;
    }

    ++l;
    --r;
  }

  return true;
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
/// 242. Valid Anagram
//------------------------------------------------------------------------------
bool ValidAnagram::is_anagram(string s, string t)
{
  if (s.size() != t.size())
  {
    return false;
  }
  // Use unordered_map for O(1) amoritized access.
  // For each letter, map it to the number of times it was seen in string s.
  // O(S) space complexity, where S is number of unique characters in s.
  std::unordered_map<char, int> letter_to_counts {};

  // O(|s|) time complexity.
  for (const char c : s)
  {
    letter_to_counts[c] += 1;
  }

  // O(|t|) time complexity.
  for (const char c : t)
  {
    if (letter_to_counts.count(c) != 1)
    {
      return false;
    }
    else
    {
      letter_to_counts[c] -= 1;
    }
  }

  for (const auto& [key, counts] : letter_to_counts)
  {
    if (letter_to_counts[key] != 0)
    {
      return false;
    }
  }

  return true;
}

//------------------------------------------------------------------------------
/// 704. Binary Search
//------------------------------------------------------------------------------
int BinarySearch::search(vector<int>& nums, int target)
{
  const int N {static_cast<int>(nums.size())};

  int l {0};
  int r {N - 1};

  while (l <= r)
  {
    const int mid { (r - l) / 2 + l};

    if (target == nums[mid])
    {
      return mid;
    }
    else if (target < nums[mid])
    {
      r = mid - 1;
    }
    else if (target > nums[mid])
    {
      l = mid + 1;
    }
  }

  return -1;
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
