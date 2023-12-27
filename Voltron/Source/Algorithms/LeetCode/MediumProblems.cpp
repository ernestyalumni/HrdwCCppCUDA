#include "MediumProblems.h"

#include <algorithm> // std::sort, std::swap;
#include <array>
#include <climits> // INT_MIN
#include <cstddef> // std::size_t
#include <limits.h>
#include <limits>
#include <map>
#include <numeric> // std::accumulate
#include <set>
#include <stack>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

using std::array;
using std::max;
using std::min;
using std::size_t;
using std::stack;
using std::string;
using std::swap;
using std::unordered_set;
using std::vector;

namespace Algorithms
{
namespace LeetCode
{

int LongestSubstringWithoutRepeating::length_of_longest_substring(string s)
{
  const int N {static_cast<int>(s.size())};

  if (N == 0)
  {
    return 0;
  }

  // We need a way to keep track of the characters we've seen.
  std::map<char, int> seen_characters {};

  int maximum_length {0};
  int initial_index_pointer {0};
  int forward_index_pointer {0};

  while (forward_index_pointer < N)
  {
    if (
      (seen_characters.count(s[forward_index_pointer]) != 0) &&
      // We sanity check if we even need to exclude the prior character.
      (seen_characters[s[forward_index_pointer]] >= initial_index_pointer))
    {
      initial_index_pointer = seen_characters[s[forward_index_pointer]] + 1;
    }

    // Mark that we've seen this character. Even if we had a duplicate, we now
    // change the index to the current index seen.
    seen_characters[s[forward_index_pointer]] = forward_index_pointer;

    maximum_length = max(
      maximum_length,
      forward_index_pointer - initial_index_pointer + 1);

    ++forward_index_pointer;
  }

  return maximum_length;
}

/// \name 5. Longest Palindromic Substring

string LongestPalindrome::brute_force(string s)
{
  const int N {static_cast<int>(s.size())};

  if (N == 1)
  {
    return s;
  }

  if (N == 2)
  {
    return is_palindrome(s) ? s : s.substr(0, 1);
  }

  int maximum_length {1};
  int maximum_length_starting_index {0};

  // O(N^2) time complexity.
  for (int i {0}; i < N; ++i)
  {
    // j index is an index that is the last index on the string, on the right.
    for (int j {i + maximum_length}; j < N; ++j)
    {
      // https://en.cppreference.com/w/cpp/string/basic_string/substr
      // Returns a substring [pos, pos + count). If requested substring extends
      // past end of string, i.e. count is greater than size() - pos, returned
      // substring is [pos, size())
      if (is_palindrome(s.substr(i, j - i + 1)))
      {
        maximum_length = j - i + 1;
        maximum_length_starting_index = i;
      }
    }
  }

  return s.substr(maximum_length_starting_index, maximum_length);
}

bool LongestPalindrome::is_palindrome(const string& s)
{
  int left {0};
  int right {static_cast<int>(s.size()) - 1};

  // O(N) time complexity.
  while (left < right)
  {
    if (s[left] != s[right])
    {
      return false;
    }
    ++left;
    --right;
  }

  return true;
}

string LongestPalindrome::expand_around_center(string s)
{
  const int N {static_cast<int>(s.size())};
  int maximum_length {1};
  int maximum_length_starting_index {0};

  // for loop is O(N) time complexity.
  for (int i {0}; i < N; ++i)
  {
    // O(N) time complexity.
    const auto odd_length_indices = expand_from_center(s, i, i);

    const int odd_length {
      std::get<1>(odd_length_indices) - std::get<0>(odd_length_indices) + 1};

    if (odd_length > maximum_length)
    {
      maximum_length = odd_length;
      maximum_length_starting_index = std::get<0>(odd_length_indices);
    }

    if ((i < N - 1) && (s[i] == s[i + 1]))
    {
      // O(N) time complexity.
      const auto even_length_indices = expand_from_center(s, i, i + 1);

      const int even_length {
        std::get<1>(even_length_indices) - std::get<0>(even_length_indices) + 1};

      if (even_length > maximum_length)
      {
        maximum_length = even_length;
        maximum_length_starting_index = std::get<0>(even_length_indices);
      }
    }
  }

  return s.substr(maximum_length_starting_index, maximum_length);
}

std::tuple<int, int> LongestPalindrome::expand_from_center(
  const std::string& s,
  const int left,
  const int right)
{
  int l {left};
  int r {right};

  // O(N) time complexity.
  while (l >= 0 && r < static_cast<int>(s.size()) && s[l] == s[r])
  {
    --l;
    ++r;
  }

  return std::make_tuple<int, int>(std::move(l + 1), std::move(r - 1));
}

string LongestPalindrome::with_dynamic_programming(std::string s)
{
  const int N {static_cast<int>(s.size())};

  // O(N^2) space complexity.
  vector<vector<bool>> is_palindrome (N, vector<bool>(N, false));

  int maximum_length {1};
  int maximum_length_starting_index {0};

  // Every single character is a palindrome.
  for (int i {0}; i < N; ++i)
  {
    is_palindrome[i][i] = true;
  }

  // Check for every substring length.
  // O(N^2) time complexity overall, with 2 for loops.
  for (int length {2}; length <= N; ++length)
  {
    for (int i {0}; i <= N - length; ++i)
    {
      int j {i + length - 1};

      // If our "outer" characters, most left and most right letters of our 
      // substring of length, are equal, and either we are checking a substring
      // of length 2, or we had a palindrome "beforehand",
      if (s[i] == s[j] && (length == 2 || is_palindrome[i + 1][j - 1]))
      {
        is_palindrome[i][j] = true;
        if (length > maximum_length)
        {
          maximum_length = length;
          maximum_length_starting_index = i;
        }
      }
    }
  }

  return s.substr(maximum_length_starting_index, maximum_length);
}

string LongestPalindrome::longest_palindrome(string s)
{
  const int N {static_cast<int>(s.size())};
  int current_index {N / 2};
  int longest_palindrome_index {current_index};
  int maximum_length {0};

  find_longest_palindrome(
    current_index,
    1,
    maximum_length,
    longest_palindrome_index,
    s,
    N);

  return s.substr(longest_palindrome_index, maximum_length);
}

void LongestPalindrome::find_longest_palindrome(
  int start_index,
  int current_length,
  int& maximum_length,
  int& longest_palindrome_index,
  const string& s,
  const int N)
{
  if (start_index < 0 || start_index + current_length - 1 >= N)
  {
    return;
  }

  if (!is_outer_letters_palindrome(start_index, current_length, s))
  {
    return;
  }
  else
  {
    if (current_length > maximum_length)
    {
      maximum_length = current_length;
      longest_palindrome_index = start_index;
    }

    find_longest_palindrome(
      start_index - 1,
      current_length + 2,
      maximum_length,
      longest_palindrome_index,
      s,
      N);
  }

  find_longest_palindrome(
    start_index - 1,
    1,
    maximum_length,
    longest_palindrome_index,
    s,
    N);

  find_longest_palindrome(
    start_index + 1,
    1,
    maximum_length,
    longest_palindrome_index,
    s,
    N);
}

/// \name 11. Container With Most Water

int ContainerWithMostWater::brute_force(vector<int>& height)
{
  const int N {static_cast<int>(height.size())};

  int maximum_area {0};

  for (int i {0}; i < N - 1; ++i)
  {
    for (int j {i + 1}; j < N; ++j)
    {
      maximum_area = std::max(
        maximum_area,
        std::min(height[i], height[j]) * (j - i));
    }
  }

  return maximum_area;
}

int ContainerWithMostWater::maximum_area(vector<int>& height)
{
  const int N {static_cast<int>(height.size())};

  // Initialize 2 points. Place 1 pointer at the beginning and the other at the
  // end.
  int l {0};
  int r {N - 1};

  int maximum_area {0};

  // If l == r, we have 0 water, 0 area.
  while (l < r)
  {
    // Calculate area, and update maximum area if current area is larger.
    int current_area {(r - l) * std::min(height[l], height[r])};

    maximum_area = std::max(maximum_area, current_area);

    // Move pointers: move pointer pointing to shorter line towards the other
    // pointer by 1 step. The rationale is that moving the shorter line might
    // lead to a taller line, potentially increasing the area. Moving the taller
    // line wouldn't increase the area (EY: possibly?)
    if (height[l] < height[r])
    {
      l++;
    }
    else
    {
      r--;
    }
  }

  return maximum_area;
}

//------------------------------------------------------------------------------
/// 15. 3Sum
//------------------------------------------------------------------------------
vector<vector<int>> ThreeSum::three_sum(vector<int>& nums)
{
  const int N {static_cast<int>(nums.size())};

  vector<vector<int>> triplets {};

  // Key insight is to first *sort the array*.
  sort(nums.begin(), nums.end());

  for (int i {0}; i < N; ++i)
  {
    const int current_complement {-nums[i]};

    int l {i + 1};
    int r {N - 1};

    while (l < r)
    {
      const int two_sum {nums[l] + nums[r]};

      if (two_sum > current_complement)
      {
        --r;
      }
      else if (two_sum < current_complement)
      {
        ++l;
      }
      else
      {
        triplets.push_back({nums[i], nums[l], nums[r]});

        // Skip past any duplicate values from the left and right.

        while ((l < r) && nums[l] == nums[l + 1])
        {
          ++l;
        }
        while ((l < r) && nums[r] == nums[r - 1])
        {
          --r;
        }

        ++l;
        --r;
      }
    }
  }

  return triplets;
}

//------------------------------------------------------------------------------
/// 33. Search in Rotated Sorted Array
//------------------------------------------------------------------------------
int SearchInRotatedSortedArray::search(vector<int>& nums, int target)
{
  const int N {static_cast<int>(nums.size())};

  if (nums.size() == 1)
  {
    return nums[0] == target ? 0 : -1;
  }

  int l {0};
  int r {N - 1};

  while (l <= r)
  {
    const int mid { (r - l) / 2 + l};

    if (nums[mid] == target)
    {
      return mid;
    }

    // Check if left side is sorted (the pivot point must be somewhere along the
    // array or no pivot point at all).
    if (nums[l] <= nums[mid])
    {
      if (target >= nums[l] && target <= nums[mid])
      {
        r = mid - 1;
      }
      else
      {
        l = mid + 1;
      }
    }
    else
    {
      if (target >= nums[mid + 1] && target <= nums[r])
      {
        l = mid + 1;
      }
      else
      {
        r = mid - 1;
      }
    }
  }

  return -1;
};

//------------------------------------------------------------------------------
/// 53. Maximum Subarray
//------------------------------------------------------------------------------
int MaximumSubarray::max_subarray(vector<int>& nums)
{
  const int N {static_cast<int>(nums.size())};

  int local_maximum {nums[0]};
  int global_maximum {nums[0]};

  for (int i {1}; i < N; ++i)
  {
    const int current_value {nums[i]};

    local_maximum = max(current_value, local_maximum + current_value);
    global_maximum = max(local_maximum, global_maximum);
  }

  return global_maximum;
}

//------------------------------------------------------------------------------
/// 56. Merge Intervals
//------------------------------------------------------------------------------
vector<vector<int>> MergeIntervals::merge(vector<vector<int>>& intervals)
{
  // Sort by first value of each interval. Then we can immediately spot there's
  // an overlap by when start_i >= end_{i-1}.
  // O(N log(N)) comparisons.
  std::sort(
    intervals.begin(),
    intervals.end(),
    [](const auto& a, const auto& b) -> bool
    {
      return a[0] < b[0];
    });

  vector<vector<int>> merged_intervals {};

  bool is_in_interval {false};
  // Constraint was start_i => 0
  int merged_start_value {-1};
  int merged_end_value {-1};
  for (const auto& interval : intervals)
  {
    if (!is_in_interval)
    {
      merged_start_value = interval[0];
      merged_end_value = interval[1];
      is_in_interval = true;
    }
    else
    {
      if (interval[0] > merged_end_value)
      {
        merged_intervals.emplace_back(
          vector<int>{merged_start_value, merged_end_value});
        merged_start_value = interval[0];
        merged_end_value = interval[1];
      }
      else if (interval[1] > merged_end_value)
      {
        merged_end_value = interval[1];
      }
    }
  }
  merged_intervals.emplace_back(
    vector<int>{merged_start_value, merged_end_value});

  return merged_intervals;
}

/// \name 75. Sort Colors

void SortColors::sort_colors(vector<int>& nums)
{
  const int N {static_cast<int>(nums.size())};
  // Index of the start of a category of a color.
  int end {0};
  int current_ptr {0};

  while (current_ptr < N && end < N)
  {
    if (nums[current_ptr] == 0)
    {
      if (end != current_ptr)
      {
        const int original_value {nums[end]};
        nums[end] = 0;
        nums[current_ptr] = original_value;
      }

      end++;
    }

    current_ptr++;
  }

  current_ptr = end;

  while (current_ptr < N && end < N)
  {
    if (nums[current_ptr] == 1)
    {
      if (end != current_ptr)
      {
        const int original_value {nums[end]};
        nums[end] = 1;
        nums[current_ptr] = original_value;
      }

      end++;
    }

    current_ptr++;
  }

  current_ptr = end;

  while (current_ptr < N && end < N)
  {
    if (nums[current_ptr] == 2)
    {
      if (end != current_ptr)
      {
        const int original_value {nums[end]};
        nums[end] = 2;
        nums[current_ptr] = original_value;
      }

      end++;
    }

    current_ptr++;
  }
}

void SortColors::one_pass(vector<int>& nums)
{
  int low_index {0};
  int current_index {0};
  int high_index {static_cast<int>(nums.size() - 1)};

  while (current_index <= high_index)
  {
    if (nums[current_index] == 0)
    {
      swap(nums[low_index], nums[current_index]);
      low_index++;
      current_index++;
    }
    else if (nums[current_index] == 1)
    {
      current_index++;
    }
    else
    {
      swap(nums[current_index], nums[high_index]);
      high_index--;
    }
  }
}

//------------------------------------------------------------------------------
/// 152. Maximum Product Subarray
//------------------------------------------------------------------------------
int MaximumProductSubarray::max_product(vector<int>& nums)
{
  const int N {static_cast<int>(nums.size())};

  int local_maximum {nums[0]};
  int local_minimum {nums[0]};
  int global_maximum {nums[0]};

  for (int i {1}; i < N; ++i)
  {
    const int current_value {nums[i]};

    if (current_value < 0)
    {
      swap(local_maximum, local_minimum);
    }    

    local_minimum = min(current_value, local_minimum * current_value);
    local_maximum = max(current_value, local_maximum * current_value);

    global_maximum = max(global_maximum, local_maximum);
  }

  return global_maximum;
}

/// \name 209. Minimum Size Subarray Sum
// Recall that a subarray is a contiguous, non-empty sequence of elements of the
// array.
int MinimumSizeSubarraySum::minimum_subarray_length(
  int target,
  vector<int>& nums)
{
  const int N {static_cast<int>(nums.size())};
  int length {INT_MAX};
  int start {0};
  // Starting with end = 0 is most logical because we're effectively starting
  // with an empty subarray at the beginning of the array.
  int end {0};
  int total_sum {0};

  // We want the subarray elements to sum equal to or greater than target.
  // We start from end = 0. We increment end until the condition above is
  // satisfied. When it's satisfied, then we try to minimize the length from the
  // "left", increment start.
  //
  // In other words, the sliding window expands to include more elements when
  // necessary and contracts to find minimum length subarray satisfying the sum
  // condition.
  while (end < N)
  {
    total_sum += nums[end];

    while ((start <= end) && (total_sum >= target))
    {
      length = min(length, end - start + 1);
      total_sum -= nums[start];
      ++start;
    }

    ++end;
  }

  // length == INT_MAX takes care of if the array's elements all of them don't
  // sum up greater than or equal to target.
  return length == INT_MAX ? 0 : length;
}

//------------------------------------------------------------------------------
/// 238. Product of Array Except self
//------------------------------------------------------------------------------

vector<int> ProductOfArrayExceptSelf::brute_force(vector<int>& nums)
{
  const int N {static_cast<int>(nums.size())};
  vector<int> products (N, 1);

  for (int i {0}; i < N; ++i)
  {
    for (int j {0}; j < N; ++j)
    {
      if (j != i)
      {
        products[j] *= nums[i];
      }
    }
  }

  return products;
}

vector<int> ProductOfArrayExceptSelf::product_except_self(vector<int>& nums)
{
  const int N {static_cast<int>(nums.size())};
  vector<int> left_products (N, 1);
  // We'll reuse products for our right products.
  vector<int> products (N, 1);

  // Left products:
  // L[i] = \begin{cases} 1 & if i == 0 \\
  //  L[i-1] * nums[i - 1] & if i > 0
  
  for (int j {1}; j < N; ++j)
  {
    left_products[j] = left_products[j - 1] * nums[j - 1];
  }

  // Right products:
  // R[i] = \begin{cases} 1 & if i == N - 1 \\
  //  R[i + 1] * nums[i + 1] & if i < N - 1

  for (int j {N - 2}; j >= 0; --j)
  {
    products[j] = products[j + 1] * nums[j + 1];
  }

  for (int i {0}; i < N; ++i)
  {
    products[i] *= left_products[i];
  }

  return products;
}

//------------------------------------------------------------------------------
/// 357. Count Numbers With Unique Digits
//------------------------------------------------------------------------------

int CountNumbersWithUniqueDigits::count_numbers_with_unique_digits(int n)
{
  /*
  if (n == 0)
  {
    return 1;
  }

  // Is this the most significant digit? (so that it cannot be of value 0)?
  bool is_first_value {true};

  array<bool, 10> is_used {false};

  // We mean by inner meaning they are digits that are not the most significant
  // digit (so it's not at the "end");
  const unordered_set<int> possible_inner_digits {1, 2, 3, 4, 5, 6, 7, 8, 9};

  stack<int> s {};
  for (int i {1}; i < 10; ++i)
  {
    s.push(i);
  }

  int counter {0};

  if (n == 1)
  {
    while (!s.empty())
    {
      s.pop();
      ++counter;
    }
  }

  while (!s.empty())
  {

  }
  */
  if (n == 0)
  {
    return 1;
  }

  if (n == 1)
  {
    return 10;
  }

  if (n == 2)
  {
    return 91;
  }

  vector<int> number_of_numbers (n + 1, 0);
  number_of_numbers.at(0) = 1;
  number_of_numbers.at(1) = 10;
  number_of_numbers.at(2) = 9 * 9 + number_of_numbers.at(1);
  number_of_numbers.at(3) = 9 * 9 * 8 + number_of_numbers.at(2);

  if (n == 3)
  {
    return number_of_numbers.at(n);
  }

  // O(n) time complexity.
  for (int i {4}; i <= n; ++i)
  {
    number_of_numbers.at(i) = number_of_numbers.at(i - 1);
    int numbers_with_length_i {9};
    int factor {9};
    // Because this doesn't depend on n, then the overall time complexity is
    // still O(n)
    for (int j {i}; j > 1; --j)
    {
      numbers_with_length_i *= factor;
      --factor;
    }
    number_of_numbers.at(i) += numbers_with_length_i;
  }

  return number_of_numbers.at(n);
}

//------------------------------------------------------------------------------
/// \name 435. Non-overlapping intervals
//------------------------------------------------------------------------------
int NonOverlappingIntervals::erase_overlap_intervals(
  vector<vector<int>>& intervals)
{
  std::sort(
    intervals.begin(),
    intervals.end(),
    [](const auto& a, const auto& b) -> bool
    {
      if (a[0] != b[0])
      {
        return a[0] < b[0];
      }
      else
      {
        return a[1] < b[1];
      }
    });

  // I'm not able to right now show definitively that this is true, even with
  // proof by contradiction, that it'll result in less number of intervals to
  // remove if we eliminate an interval that overlaps with more than 2 intervals
  // (or by definition, that's the case).

  int current_start_value {INT_MIN};
  int current_end_value {INT_MIN};
  int counter {0};

  for (const auto& interval : intervals)
  {
    // Overlap detected.
    if (interval[0] < current_end_value)
    {
      ++counter;
      // TODO: This step isn't mathematically proven definitively that seeking
      // smaller intevals will lead to less intervals to remove.
      if (interval[1] < current_end_value)
      {
        current_start_value = interval[0];
        current_end_value = interval[1];
      }
    }
    // No overlap detected, or we start to track an interval for overlaps in the
    // next iteration.
    else
    {
      current_start_value = interval[0];
      current_end_value = interval[1];
    }
  }

  return counter;
}

//------------------------------------------------------------------------------
/// \name 547. Number of Provinces
/// \url https://leetcode.com/problems/number-of-provinces/
//------------------------------------------------------------------------------
int find_number_of_provinces(vector<vector<int>>& is_connected)
{
  const size_t n {is_connected.size()};

  if (n == 0)
  {
    return static_cast<int>(n);
  }

  size_t counter {0}; 

  for (size_t i {0}; i < n; ++i)
  {
    vector<int>& row_to_consider {is_connected[i]};

    for (size_t j {i + 1}; j < n; ++j)
    {
      // There's a direct connection.
      if (row_to_consider[j] == 1)
      {
        is_connected[j][i] = 0;
        // Mark that this city is connected.
        is_connected[j][j] = 0;
      }
    }
    
    counter += is_connected[i][i];
  }

  return counter;
}

/// \name 647. Palindromic Substrings

int PalindromicSubstrings::brute_force(string s)
{
  const int N {static_cast<int>(s.size())};
  int number_of_palindromes {0};

  // O(N^2) time complexity overall.
  for (int i {0}; i < N; ++i)
  {
    for (int j {i}; j < N; ++j)
    {
      if (is_palindrome(s.substr(i, j - i + 1)))
      {
        number_of_palindromes += 1;
      }
    }
  }

  return number_of_palindromes;
}

int PalindromicSubstrings::count_substrings(string s)
{
  const int N {static_cast<int>(s.size())};
  // Number of palindromic substrings.
  int count {0};

  // Expand from the center with the center being from both each element of
  // string s and each adjacent pairs of elements.
  // Stop the expansion when it's no longer a palindrome!
  for (int i {0}; i < 2 * N - 1; ++i)
  {
    int left {i / 2};
    int right {left + i % 2};

    while ((0 <= left && right < N) && (s[left] == s[right]))
    {
      ++count;
      --left;
      ++right;
    }
  }

  return count;
}

bool PalindromicSubstrings::is_palindrome(const string& s)
{
  const int N {static_cast<int>(s.size())};

  int start {0};
  int end {N - 1};

  while (start < end)
  {
    if (s[start] != s[end])
    {
      return false;
    }

    ++start;
    --end;
  }

  return true;
}

/// \name 739. Daily Temperatures

vector<int> DailyTemperatures::brute_force(vector<int>& temperatures)
{
  const int N {static_cast<int>(temperatures.size())};
  vector<int> answer (N, 0);

  for (int i {0}; i < N - 1; ++i)
  {
    int days_to_warmer_temperature {0};
    const int current_temperature {temperatures[i]};
    bool is_warmer_temperature_found {false};
    for (int j {i + 1}; j < N; ++j)
    {
      ++days_to_warmer_temperature;
      if (temperatures[j] > current_temperature)
      {
        is_warmer_temperature_found = true;
        break;
      }
    }

    if (is_warmer_temperature_found)
    {
      answer[i] = days_to_warmer_temperature;
    }
    else
    {
      answer[i] = 0;
    }
  }

  return answer;
}

vector<int> DailyTemperatures::daily_temperatures(vector<int>& temperatures)
{
  // Store indices for temperatures that haven't found a subsequent warmer day.
  stack<int> colder_temperatures {};

  const int N {static_cast<int>(temperatures.size())};
  vector<int> answer (N, 0);

  for (int i {0}; i < N; ++i)
  {
    if (!colder_temperatures.empty())
    {
      bool is_previous_temperature_warmer {false};

      while (!is_previous_temperature_warmer && !colder_temperatures.empty())
      {
        const int previous_index {colder_temperatures.top()};

        if (temperatures[previous_index] < temperatures[i])
        {
          answer[previous_index] = i - previous_index;
          colder_temperatures.pop();
        }
        else
        {
          is_previous_temperature_warmer = true;
        }
      }
    }

    colder_temperatures.push(i);
  }

  // Handle the remaining stack - if stack still has indices after iterating
  // through array, it means those days don't have warmer future temperature.
  // Mark as 0 in result array.
  while (!colder_temperatures.empty())
  {
    answer[colder_temperatures.top()] = 0;
    colder_temperatures.pop();
  }

  return answer;
}

/// \name 2944. Minimum Number of Coins for Fruits
int MinimumNumberOfCoinsForFruits::minimum_coins(vector<int>& prices)
{
  // N number of fruits (each of different types)
  const int N {static_cast<int>(prices.size())};

  // Minimum cost to purchase the first i fruits for the ith element.
  vector<int> minimum_cost (N + 1, std::numeric_limits<int>::max());
  minimum_cost[0] = 0;
  // No.
  // This is the base case as you "must" purchase the first fruit.
  //minimum_cost[1] = 0;

  // i is for ith fruit with fruits begin 1-indexed.
  for (int i {1}; i <= N; ++i)
  {
    // Consider buying the ith fruit and getting the next i fruits for free.
    // Fruits you get for free after buying fruit i. j is 1-indexed.
    for (int j {i}; j <= std::min(N, i + i); ++j)
    {
      minimum_cost[j] = std::min(
        minimum_cost[j],
        minimum_cost[i - 1] + prices[i - 1]);
    }
  }

  return minimum_cost[N];
}

} // namespace LeetCode
} // namespace Algorithms
