#include "MediumProblems.h"

#include "DataStructures/BinaryTrees.h"

#include <algorithm> // std::sort, std::swap;
#include <array>
#include <climits> // INT_MIN
#include <cstddef> // std::size_t
#include <cstdint>
#include <functional>
#include <iomanip> // std::setfill, std::setw
#include <limits.h>
#include <limits>
#include <map>
#include <numeric> // std::accumulate
#include <optional>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility> // std::pair, std::make_pair
#include <vector>

using DataStructures::BinaryTrees::TreeNode;
using std::array;
using std::function;
using std::get;
using std::make_pair;
using std::make_tuple;
using std::max;
using std::min;
using std::move;
using std::pair;
using std::priority_queue;
using std::queue;
using std::size_t;
using std::sort;
using std::stack;
using std::string;
using std::stringstream;
using std::swap;
using std::to_string;
using std::tuple;
using std::unordered_map;
using std::unordered_set;
using std::vector;

namespace Algorithms
{
namespace LeetCode
{

//------------------------------------------------------------------------------
/// 3. Longest Substring Without Repeating Characters
/// https://leetcode.com/problems/longest-substring-without-repeating-characters/description/
/// Constraints:
///
/// 0 <= s.length <= 5 * 104
/// s consists of English letters, digits, symbols and spaces.
//------------------------------------------------------------------------------

int LongestSubstringWithoutRepeating::length_of_longest_substring(string s)
{
  const int N {static_cast<int>(s.size())};

  if (N == 0)
  {
    return 0;
  }

  // We need a way to keep track of the characters we've seen.
  // std::map is typically O(log n)
  // character seen to index of the character in the string.
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

int LongestSubstringWithoutRepeating::concise_length_of_longest_substring(
  string s)
{
  const int N {static_cast<int>(s.size())};
  if (N == 0)
  {
    return 0;
  }

  // Keep track of the characters we've seen using std::unordered_set for
  // O(1) amoritized access.
  unordered_map<char, int> seen_characters {};

  int length {0};
  int start {0};

  for (int j {0}; j < N; ++j)
  {
    const char current_c {s[j]};

    // Seen character before.
    if (seen_characters.count(current_c) != 0)
    {
      // You need to take the max because you could've seen the character before
      // the beginning of this substring. Consider "tmmzuxt" and the case of the
      // second "t."
      start = max(start, seen_characters[current_c] + 1);
    }

    seen_characters[current_c] = j;

    length = max(length, j - start + 1);
  }

  return length;
}

//------------------------------------------------------------------------------
/// \name 5. Longest Palindromic Substring
//------------------------------------------------------------------------------

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
  // O(N log N) complexity.
  sort(nums.begin(), nums.end());

  for (int i {0}; i < N; ++i)
  {
    // Skip duplicate fixed element values.
    if (i > 0 && nums[i - 1] == nums[i])
    {
      continue;
    }

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
      // Since the left side is sorted in ascending order, check if the target
      // is in here or not.
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
/// 48. Rotate Image
//------------------------------------------------------------------------------
void RotateImage::rotate(vector<vector<int>>& matrix)
{
  const int N {static_cast<int>(matrix.size())};

  if (N == 1)
  {
    return;
  }

  // N is even.
  int T {N / 2 - 1};
  int B {N / 2};
  int L {N / 2 - 1};
  int R {N / 2};

  // If N odd.
  if (N % 2 == 1)
  {
    T = N / 2 - 1;
    B = N / 2 + 1;
    L = N / 2 - 1;
    R = N / 2 + 1;
  }

  while (T >= 0 && (B <= N - 1) && (L >= 0) && (R <= N - 1))
  {
    // We'll move around clockwise in a spiral in considering what 4 elements to
    // swap in a square.
    int top_left {L};
    int top_right {T};
    int bottom_right {R};
    int bottom_left {B};

    while (
      top_left < R &&
      top_right < B &&
      bottom_right > L &&
      bottom_left > T)
    {
      swap(matrix[T][top_left], matrix[bottom_left][L]);
      swap(matrix[bottom_left][L], matrix[B][bottom_right]);
      swap(matrix[B][bottom_right], matrix[top_right][R]);

      // Rotate the indices clockwise.
      ++top_left;
      ++top_right;
      --bottom_right;
      --bottom_left;
    }

    // Update the spiral outward.
    T--;
    B++;
    L--;
    R++;
  }
}


//------------------------------------------------------------------------------
/// 49. Group Anagrams
/// https://leetcode.com/problems/group-anagrams/description/
/// https://neetcode.io/problems/anagram-groups
//------------------------------------------------------------------------------

// TODO: Doesn't work, not sure why? :(
/*
vector<vector<string>> GroupAnagrams::group_anagrams(vector<string>& strs)
{
  unordered_map<string, vector<string>> str_to_anagrams {};

  for (const string& str : strs)
  {
    if (str_to_anagrams.count(str) != 0)
    {
      str_to_anagrams[str].push_back(str);
    }
    else
    {
      for (const auto& [key, _] : str_to_anagrams)
      {
        if (GroupAnagrams::is_anagram_open_ai(key, str))
        {
          str_to_anagrams[key].push_back(str);
        }
      }

      str_to_anagrams[str] = {};
    }
  }

  std::cout << "\n str_to_anagrams.size(): " << str_to_anagrams.size() << "\n";

  vector<vector<string>> results {};

  for (const auto& [key, _] : str_to_anagrams)
  {
    str_to_anagrams[key].push_back(key);

    results.push_back(str_to_anagrams[key]);
  }

  return results;
}
*/

vector<vector<string>> GroupAnagrams::group_anagrams_by_sorting(
  vector<string>& strs)
{
  unordered_map<string, vector<string>> anagram_to_group {};

  // For each string, make a copy, sort it so letters in order, and use that as
  // a classification for what anagram it is, what anagram group it belongs to.
  for (const string& str : strs)
  {
    string sorted_string = str;
    sort(sorted_string.begin(), sorted_string.end());
    anagram_to_group[sorted_string].push_back(str);
  }

  vector<vector<string>> results {};

  for (auto& [_, group] : anagram_to_group)
  {
    results.push_back(group);
  }

  return results;
}

vector<vector<string>> GroupAnagrams::group_anagrams_by_frequency(
  vector<string>& strs)
{
  unordered_map<string, vector<string>> frequency_count_to_group {};

  for (const string& str : strs)
  {
    array<int, 26> frequency_count {};
    for (const char c : str)
    {
      frequency_count[static_cast<int>(c - 'a')]++;
    }

    string key {};

    // Transform frequency_count into a string
    for (int i {0}; i < 26; ++i)
    {
      // "#" added to avoid collisions.
      key += "#" + to_string(frequency_count[i]); 
    }

    frequency_count_to_group[key].push_back(str);
  }

  vector<vector<string>> results {};

  for (auto& [_, group] : frequency_count_to_group)
  {
    results.push_back(group);
  }

  return results;
}

bool GroupAnagrams::is_anagram_open_ai(const string& s, const string& t)
{
  if (s.size() != t.size())
  {
    return false;
  }

  unordered_map<char, int> letter_to_count {};

  // O(|s|) time.
  for (size_t i {0}; i < s.size(); ++i)
  {
    letter_to_count[s[i]]++;
    letter_to_count[t[i]]--;
  }

  // Check if all values for each letter is zero (that would've shown up each
  // letter of s, t).
  for (const auto& [letter, count] : letter_to_count)
  {
    if (count != 0)
    {
      return false;
    }
  }

  return true;
}

bool GroupAnagrams::is_anagram_neet_code(const string& s, const string& t)
{
  if (s.size() != t.size())
  {
    return false;
  }

  unordered_map<char, int> s_letter_to_count {};
  unordered_map<char, int> t_letter_to_count {};

  // O(|s|) time.
  for (size_t i {0}; i < s.size(); ++i)
  {
    if (s_letter_to_count.count(s[i]) != 0)
    {
      s_letter_to_count[s[i]] += 1;
    }
    else
    {
      s_letter_to_count[s[i]] = 1;
    }
    if (t_letter_to_count.count(t[i]) != 0)
    {
      t_letter_to_count[t[i]] += 1;
    }
    else
    {
      t_letter_to_count[t[i]] = 1;
    }
  }

  for (const char c : s)
  {
    if (s_letter_to_count.count(c) != t_letter_to_count.count(c))
    {
      return false;
    }
  }

  return true;
}

// https://en.cppreference.com/w/cpp/utility/hash/operator()
// std::hash<Key>::operator()
// Takes a single argument key of type Key
// returns value of type std::size_t that represents the hash value of key.


//------------------------------------------------------------------------------
/// 53. Maximum Subarray
/// Use 2 single int values, local maximum, global maximum to track
//------------------------------------------------------------------------------
int MaximumSubarray::max_subarray(vector<int>& nums)
{
  const int N {static_cast<int>(nums.size())};

  int local_maximum {nums[0]};
  int global_maximum {nums[0]};

  for (int i {1}; i < N; ++i)
  {
    const int current_value {nums[i]};

    // As we increment forward, any previous summation, local_maximum cannot add
    // "more" to any future sum, and so eliminate it by max.
    local_maximum = max(current_value, local_maximum + current_value);
    global_maximum = max(local_maximum, global_maximum);
  }

  return global_maximum;
}

//------------------------------------------------------------------------------
/// 54. Spiral Matrix
//------------------------------------------------------------------------------
vector<int> SpiralMatrix::spiral_order(vector<vector<int>>& matrix)
{
  const int M {static_cast<int>(matrix.size())};
  const int N {static_cast<int>(matrix[0].size())};

  int top {0};
  // Left most column to traverse.
  int l {0};
  // Right most column to traverse.
  int r {N - 1};
  // "lowest" row to traverse.
  int bottom {M - 1};

  vector<int> spiral_order {};

  while (top <= bottom && l <= r)
  {
    for (int j {l}; j <= r; ++j)
    {
      spiral_order.emplace_back(matrix[top][j]);
    }
    top++;

    for (int i {top}; i <= bottom; ++i)
    {
      spiral_order.emplace_back(matrix[i][r]);
    }
    r--;

    // If we don't check this, then we could possibly add an element twice.
    // After we've moved right and down, we need to move left. But if the matrix
    // was a single row, l and r may still indicate valid elements.
    if (top <= bottom)
    {
      for (int j {r}; j >= l; --j)
      {
        spiral_order.emplace_back(matrix[bottom][j]);
      }
      bottom--;
    }

    // If we don't check this, then we oculd possibly add an element twice.
    // After moving left, we have to move up. But if matrix was a single column,
    // top and bottom may still indicate valid elements as they've only been
    // incremented or decremented once.
    if (l <= r)
    {
      for (int i {bottom}; i >= top; --i)
      {
        spiral_order.emplace_back(matrix[i][l]);
      }
      l++;
    }
  }

  return spiral_order;
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

//------------------------------------------------------------------------------
/// 57. Insert Interval
//------------------------------------------------------------------------------
size_t InsertInterval::binary_search(
  vector<vector<int>>& intervals,
  vector<int>& new_interval)
{
  size_t l {0};
  size_t r {intervals.size() - 1};

  const int target_value {new_interval[0]};

  while (l < r)
  {
    const size_t midpoint_index {(r + l) / 2};
    const int midpoint_value {intervals[midpoint_index][0]};

    if (target_value == midpoint_value)
    {
      return midpoint_index;
    }

    if (target_value < midpoint_value)
    {
      r = midpoint_index;
    }
    else
    {
      l = midpoint_index + 1;
    }
  }

  return l;
}

/*
vector<vector<int>> InsertInterval::insert_with_binary_search(
  vector<vector<int>>& intervals,
  vector<int>& new_interval)
{
  // O(1) time.
  auto is_overlapping = [](const vector<int>& a, const vector<int>& b) -> bool
  {
    return (a[1] < b[0] || b[1] < a[0]) ? false : true;
  };

  // O(1) time.
  auto merge_overlapping_intervals = [](
    const vector<int>& a,
    const vector<int>& b) -> vector<int>
  {
    const int l {a[0] <= b[0] ? a[0] : b[0]};
    const int r {a[1] <= b[1] ? b[1] : a[1]};
    return vector<int>{l, r};
  };

  // O(log(N)) time.
  size_t new_place {
    InsertInterval::binary_search(intervals, new_interval)};

  vector<vector<int>> result (intervals.begin(), intervals.begin() + new_place);

  for (size_t i {new_place}; i < intervals.size(); ++i)
  {
    if (!is_overlapping(intervals[i], new_interval))
    {
      if (new_interval[0] < intervals[i][0])
      {
        result.push_back(new_interval);
        new_place = i;
        break;
      }
      result.push_back(intervals[i]);
    }
    else
    {
      new_interval = merge_overlapping_intervals(intervals[i], new_interval);
    }
  }

  if (result.empty() || result.back()[1] < new_interval[0])
  {
    result.push_back(new_interval);
  }
  else
  {
    result.back() = merge_overlapping_intervals(result.back(), new_interval);
  }

  result.insert(result.end(), intervals.begin() + new_place, intervals.end());

  return result;
}
*/

vector<vector<int>> InsertInterval::insert(
  vector<vector<int>>& intervals,
  vector<int>& new_interval)
{
  vector<vector<int>> result {};

  for (const auto& interval : intervals)
  {
    // No overlap, and new_interval is to be placed after current interval.
    if (interval[1] < new_interval[0])
    {
      result.push_back(interval);
    }
    // No overlap, and new_interval is to be placed here.
    else if (interval[0] > new_interval[1])
    {
      result.push_back(new_interval);
      // Assign the remaining intervals to be added.
      new_interval = interval;
    }
    // Overlap, and so merge into a new_interval.
    else
    {
      new_interval[0] = std::min(new_interval[0], interval[0]);
      new_interval[1] = std::max(new_interval[1], interval[1]);
    }
  }
  result.push_back(new_interval);
  return result;
}

//------------------------------------------------------------------------------
/// 73. Set Matrix Zeroes
//------------------------------------------------------------------------------
void SetMatrixZeroes::brute_force(vector<vector<int>>& matrix)
{
  // For a given maximum of m, n <= 200, then 200 * 200 = 40000. There are 625
  // numbers of 64 bits each that can each 40000 bits.
  const int number_of_bits {64};
  array<uint64_t, 625> is_zero {};

  const int M {static_cast<int>(matrix.size())};
  const int N {static_cast<int>(matrix[0].size())};

  // O(MN) time complexity.
  for (int i {0}; i < M; ++i)
  {
    for (int j {0}; j < N; ++j)
    {
      if (matrix[i][j] == 0)
      {
        const int k {i * N + j};
        is_zero[k / number_of_bits] |=
          (static_cast<uint64_t>(1) << (k % number_of_bits));
      }
    }
  }

  for (int k {0}; k < M * N; ++k)
  {
    if (static_cast<bool>(is_zero[k / number_of_bits] & 
      (static_cast<uint64_t>(1) << (k % number_of_bits))))
    {
      const int I {k / N};
      const int J {k % N};

      for (int j {0}; j < N; ++j)
      {
        matrix[I][j] = 0;
      }

      for (int i {0}; i < M; ++i)
      {
        matrix[i][J] = 0;
      }
    }
  }
}

void SetMatrixZeroes::set_zeroes(vector<vector<int>>& matrix)
{
  const int number_of_bits {64};
  // Track which rows will have all zeroes.
  array<uint64_t, 4> is_row_zeroes {};
  // Track which columns will have all zeroes.
  array<uint64_t, 4> is_column_zeroes {};

  const int M {static_cast<int>(matrix.size())};
  const int N {static_cast<int>(matrix[0].size())};

  // O(MN) time complexity.
  for (int i {0}; i < M; ++i)
  {
    for (int j {0}; j < N; ++j)
    {
      if (matrix[i][j] == 0)
      {
        is_row_zeroes[i / number_of_bits] |=
          (static_cast<uint64_t>(1) << (i % number_of_bits));

        is_column_zeroes[j / number_of_bits] |=
          (static_cast<uint64_t>(1) << (j % number_of_bits));
      }
    }
  }

  for (int i {0}; i < M; ++i)
  {
    if (static_cast<bool>(is_row_zeroes[i / number_of_bits] &
      (static_cast<uint64_t>(1) << (i % number_of_bits))))
    {
      for (int j {0}; j < N; ++j)
      {
        matrix[i][j] = 0;
      }
    }
  }

  for (int j {0}; j < N; ++j)
  {
    if (static_cast<bool>(is_column_zeroes[j / number_of_bits] &
      (static_cast<uint64_t>(1) << (j % number_of_bits))))
    {
      for (int i {0}; i < M; ++i)
      {
        matrix[i][j] = 0;
      }
    }
  }
}

//------------------------------------------------------------------------------
/// 74. Search a 2D Matrix
//------------------------------------------------------------------------------

bool SearchA2DMatrix::search_matrix(vector<vector<int>>& matrix, int target)
{
  const int M {static_cast<int>(matrix.size())};
  const int N {static_cast<int>(matrix[0].size())};

  int l {0};
  int r {M * N - 1};

  while (l <= r)
  {
    const int mid {l + (r - l) / 2};

    const int current_value {matrix[mid / N][mid % N]};

    if (current_value == target)
    {
      return true;
    }
    else if (current_value > target)
    {
      r = mid - 1;
    }
    else
    {
      l = mid + 1;
    }
  }

  return false;
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
/// 98. Validate Binary Search Tree
//------------------------------------------------------------------------------

bool ValidateBinarySearchTree::is_valid_BST(TreeNode* root)
{
  function<bool(TreeNode*, const long long, const long long)> step = [&](
    TreeNode* node,
    const long long lower_bound,
    const long long upper_bound) -> bool
  {
    // We reached the end of a node and it's legitmate to have no child.
    if (node == nullptr)
    {
      return true;
    }

    const int current_value {node->value_};

    if (static_cast<long long>(current_value) <= lower_bound ||
      static_cast<long long>(current_value) >= upper_bound)
    {
      return false;
    }

    if (node->left_ != nullptr && node->left_->value_ >= current_value)
    {
      return false;
    }
    else if (node->right_ != nullptr &&
      node->right_->value_ <= current_value)
    {
      return false;
    }

    return step(
      node->left_, lower_bound, static_cast<long long>(current_value)) &&
      step(node->right_, static_cast<long long>(current_value), upper_bound);
  };

  return step(root, LONG_MIN, LONG_MAX);
}

bool ValidateBinarySearchTree::is_valid_BST_track_parent_pointer(TreeNode* root)
{
  function<bool(TreeNode*, TreeNode*, TreeNode*)> step = [&](
    TreeNode* l,
    TreeNode* node,
    TreeNode* r)
  {
    // Base case, we've reached the end of a node.
    if (node == nullptr)
    {
      return true;
    }

    if ((l != nullptr && l->value_ >= node->value_) ||
      (r != nullptr && r->value_ <= node->value_))
    {
      return false;
    }

    return step(l, node->left_, node) && step(node, node->right_, r);
  };

  return step(nullptr, root, nullptr);
}

//------------------------------------------------------------------------------
/// 102. Binary Tree Level Order Traversal
//------------------------------------------------------------------------------

// Also known as preorder or breadth-first traversal.

vector<vector<int>> BinaryTreeLevelOrderTraversal::level_order_iterative(
  TreeNode* root)
{
  vector<vector<int>> traversed_levels {};

  if (root == nullptr)
  {
    return traversed_levels;
  }

  queue<TreeNode*> level_nodes {};

  level_nodes.push(root);

  while (!level_nodes.empty())
  {
    const int level_size {static_cast<int>(level_nodes.size())};

    vector<int> traversed_nodes {};

    for (int i {0}; i < level_size; ++i)
    {
      // FIFO - first in, first out.
      traversed_nodes.emplace_back(level_nodes.front()->value_);

      if (level_nodes.front()->left_ != nullptr)
      {
        level_nodes.push(level_nodes.front()->left_);
      }

      if (level_nodes.front()->right_ != nullptr)
      {
        level_nodes.push(level_nodes.front()->right_);
      }

      level_nodes.pop();
    }

    traversed_levels.emplace_back(traversed_nodes);
  }

  return traversed_levels;
}

vector<vector<int>> BinaryTreeLevelOrderTraversal::level_order_recursive(
  TreeNode* root)
{
  int level {0};
  vector<vector<int>> traversed {};

  function<void(TreeNode*, const int)> step = [&](
    TreeNode* node,
    const int level)
  {
    // We've reached a leaf's child. Return to get out of this "stack."
    if (node == nullptr)
    {
      return;
    }

    // Ensure the results or i.e. traversed vector is large enough to hold the
    // current level and be able to access with operator[].
    if (level >= static_cast<int>(traversed.size()))
    {
      for (int i {0}; i < (level - static_cast<int>(traversed.size()) + 1); ++i)
      {
        traversed.emplace_back(vector<int>{});
      }
    }

    traversed[level].emplace_back(node->value_);

    step(node->left_, level + 1);
    step(node->right_, level + 1);
  };

  step(root, 0);

  return traversed;
}

//------------------------------------------------------------------------------
/// 128. Longest Consecutive Sequence
//------------------------------------------------------------------------------
int LongestConsecutiveSequence::longest_consecutive_with_set(vector<int>& nums)
{
  unordered_set<int> nums_as_set {};

  // O(N) time.
  for (const int num : nums)
  {
    // O(1) amoritized to insert.
    nums_as_set.insert(num);
  }

  int max_length {0};

  // O(N) time.
  for (const int num : nums)
  {
    if (max_length == 0)
    {
      max_length = 1;
    }

    // Don't start to count sequence length if element is not a "right end" such
    // that it has a right neighbor.
    if (nums_as_set.count(num + 1) != 0)
    {
      continue;
    }

    int current_length {1};
    int current_value {num};

    bool is_consecutive {true};

    // O(1) amoritized.
    while (is_consecutive)
    {
      // O(1) amoritized to find a key in a set.
      if (nums_as_set.count(current_value - 1) != 0)
      {
        current_length += 1;
        current_value -= 1;
      }
      else
      {
        is_consecutive = false;
        break;
      }
    }

    max_length = max_length < current_length ? current_length : max_length;
  }

  return max_length;
}

//------------------------------------------------------------------------------
/// 137. Single Number II
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
/// For each bit, if an element appears 3 times, the ith bit with have a sum of
/// 3. 3 % 3 = 0. But for one element, appearing exactly once, ith bit could
/// possibly be only once. After modulo 3 arithmetic on sum, reconstitute the
/// bits. 
//------------------------------------------------------------------------------
int SingleNumberII::single_number_count_per_bit(vector<int>& nums)
{
  // Use the 32 bits to store bit values.
  uint32_t result {0};

  static constexpr int NUMBER_OF_BITS {32};
  for (int i {0}; i < NUMBER_OF_BITS; ++i)
  {
    const uint32_t mask {1u << i};

    int ith_count {0};

    for (const auto num : nums)
    {
      const uint32_t value {(num & mask) >> i};
      ith_count += value;
    }

    const int ith_value {ith_count % 3};
    if (ith_value == 1)
    {
      result |= (1u << i);
    }
  }

  return static_cast<int>(result);
}

//------------------------------------------------------------------------------
/// Take advantage of XOR's property for self-inverse. For number n, if seen the
/// first time, then XOR into some number a, a ^ n. If seen a second time, then
/// XOR into some number b, b ^ n, while for a, (n ^ n) = 0. Observe that we can
/// exclude numbers in b from a and vice versa, by bitwise and & and complement.
//------------------------------------------------------------------------------
int SingleNumberII::single_number_track_seen(vector<int>& nums)
{
  int first_seen {0};
  int second_seen {0};

  for (const auto num : nums)
  {
    (first_seen ^= num) &= (~second_seen);
    (second_seen ^= num) &= (~first_seen);
  }

  return first_seen;
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

//------------------------------------------------------------------------------
/// 153. Find Minimum in Rotated Sorted Array
//------------------------------------------------------------------------------
int FindMinimumInRotatedSortedArray::find_min(vector<int>& nums)
{
  const int N {static_cast<int>(nums.size())};

  int l {0};
  int r {N - 1};

  int minimum {INT_MAX};

  while (l <= r)
  {
    const int mid {l + (r - l) / 2};

    minimum = min(minimum, nums[mid]);

    // Check if the left side is sorted in ascending order.
    // <= is needed because of how we choose mid to be floor. The entire left
    // side might be just a single value but it's still in ascending order. See
    // Test case 144, nums = {2,3,4,5,6,7,8,9,1}
    if (mid > 0 && nums[l] <= nums[mid - 1])
    {
      minimum = min(minimum, nums[l]);
      l = mid + 1;
    }
    else
    {
      if (mid < N - 1)
      {
        minimum = min(minimum, nums[mid + 1]);
      }
      r = mid - 1;
    }
  }

  return minimum;
}

//------------------------------------------------------------------------------
/// 155. Min Stack
//------------------------------------------------------------------------------
MinStack::MinStack():
  array_(MAXIMUM_CALLS_),
  // Use -1 to indicate empty stack.
  top_index_{-1},
  minimum_value_and_count_{std::nullopt}
{}

void MinStack::push(int val)
{
  if (is_empty())
  {
    minimum_value_and_count_ = {val, 1};
  }

  ++top_index_;

  array_[top_index_] = val;
}

bool MinStack::is_empty()
{
  return top_index_ == -1;
}

void MinStack::pop()
{
  if (is_empty())
  {
    return;
  }

  --top_index_;

  // Heap?
}

//------------------------------------------------------------------------------
/// 167. Two Sum II - Input Array Is Sorted
/// Return the indices of the two numbers, index1 and index2, added by one as an
/// integer array [index1, index2] of length 2.
//------------------------------------------------------------------------------
vector<int> TwoSumII::two_sum(vector<int>& numbers, int target)
{
  int l {0};
  int r {static_cast<int>(numbers.size() - 1)};

  while (l < r)
  {
    int current_sum {numbers[l] + numbers[r]};
    if (current_sum == target)
    {
      return {l + 1, r + 1};
    }

    if (current_sum < target)
    {
      // Take advantage of the fact that array is already sorted in
      // non-decreasing order so we can try to increase the sum by moving up the
      // choice of "left" element.
      l++;
    }
    else
    {
      r--;
    }
  }

  return {};
}

//------------------------------------------------------------------------------
/// 187. Repeated DNA Sequences
/// https://leetcode.com/problems/repeated-dna-sequences/description/
//------------------------------------------------------------------------------
vector<string> RepeatedDNASequences::find_repeated_dna_sequences(string s)
{
  static constexpr int DNA_SEQUENCE_LENGTH {10};
  const int N {static_cast<int>(s.size())};

  if (N <= DNA_SEQUENCE_LENGTH)
  {
    return {};
  }

  unordered_map<string, int> substring_to_count {};

  for (int i {0}; i < (N - DNA_SEQUENCE_LENGTH + 1); ++i)
  {
    const string substring {s.substr(i, DNA_SEQUENCE_LENGTH)};
    substring_to_count[substring]++;
  }

  vector<string> repeated_dna_sequences {};
  for (const auto& [substring, count] : substring_to_count)
  {
    if (count > 1)
    {
      repeated_dna_sequences.push_back(substring);
    }
  }

  return repeated_dna_sequences;
}

//------------------------------------------------------------------------------
/// 200. Number of Islands
//------------------------------------------------------------------------------

int NumberOfIslands::number_of_islands_with_depth_first_search(
  vector<vector<char>>& grid)
{

  function<void(vector<vector<char>>&, const int, const int)>
    search_for_an_island = [&](
      vector<vector<char>>& grid,
      const int i,
      const int j)
  {
    const int M {static_cast<int>(grid.size())};
    const int N {static_cast<int>(grid[0].size())};

    // If we hit a boundary (which is water from problem), or hit water, we can
    // end this traversal because we're out of an island.
    if ((i < 0) || (i >= M) || (j < 0) || (j >= N) || grid[i][j] == '0')
    {
      return;
    }

    // Key insight: mark as 0 because we've visited this here. The condition
    // above allows us not to visit this element again.
    grid[i][j] = '0';

    search_for_an_island(grid, i + 1, j);
    search_for_an_island(grid, i - 1, j);
    search_for_an_island(grid, i, j + 1);
    search_for_an_island(grid, i, j - 1);

    return;
  };

  const int M {static_cast<int>(grid.size())};
  const int N {static_cast<int>(grid[0].size())};
  int count {0};

  for (int i {0}; i < M; ++i)
  {
    for (int j {0}; j < N; ++j)
    {
      if (grid[i][j] == '1')
      {
        search_for_an_island(grid, i, j);
        ++count;
      }
    }
  }

  return count;
}

int NumberOfIslands::number_of_islands_with_breadth_first_search(
  vector<vector<char>>& grid)
{
  const int M {static_cast<int>(grid.size())};
  const int N {static_cast<int>(grid[0].size())};

  auto is_valid = [&](const int i, const int j)
  {
    return (i >= 0 && i < M && 0 <= j && j < N && grid[i][j] != '0');
  };

  int count {0};

  queue<pair<int, int>> unvisited_cells {};

  for (int i {0}; i < M; ++i)
  {
    for (int j {0}; j < N; ++j)
    {
      if (grid[i][j] != '0')
      {
        unvisited_cells.push(make_pair(i, j));
        // Mark as visited immediately, to avoid processing a cell multiple
        // times.
        grid[i][j] = '0';

        while (!unvisited_cells.empty())
        {
          //const int level_size {static_cast<int>(unvisited_cells.size())};

          //for (int element {0}; element < level_size; ++element)
          //{
            const auto ij = unvisited_cells.front();
            const int I {get<0>(ij)};
            const int J {get<1>(ij)};
            unvisited_cells.pop();

            // Remember to check if (i, j) are within the bounds.

            if (is_valid(I + 1, J))
            {
              unvisited_cells.push(make_pair(I + 1, J));
              // Mark immediately as visited.
              grid[I + 1][J] = '0';
            }
            if (is_valid(I - 1, J))
            {
              unvisited_cells.push(make_pair(I - 1, J));
              grid[I - 1][J] = '0';
            }
            if (is_valid(I, J + 1))
            {
              unvisited_cells.push(make_pair(I, J + 1));
              grid[I][J + 1] = '0';
            }
            if (is_valid(I, J - 1))
            {
              unvisited_cells.push(make_pair(I, J - 1));
              grid[I][J - 1] = '0';
            }
          //}
        }

        // We completed tracing through 1 island.
        count++;
      }
    }
  }

  return count;
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
/// 201. Bitwise AND of Numbers Range
//------------------------------------------------------------------------------

int BitwiseANDOfNumbersRange::naive_loop(int left, int right)
{
  if (left == right)
  {
    return left;
  }

  const int increment {right > left ? 1 : -1};

  int result {left};

  // O(n) time complexity.
  for (int k {left + increment}; k != right; k += increment)
  {
    result &= k;
  }

  result &= right;

  return result;
}

//------------------------------------------------------------------------------
/// Key insight: because of the numbers in between left, and right, which toggle
/// between 0 and 1 in bits lower than the msb's (most significant bit) common
/// to both left and right, we need to just get those common bits.
/// e.g. 5 (101), 7 (111)
/// We want to get the msb's that are common to both left and right, the common
/// and highest prefix.
//------------------------------------------------------------------------------
int BitwiseANDOfNumbersRange::range_bitwise_and(int left, int right)
{
  // Continue to shift left and right down until they are equal and then restore
  // the original value.
  int shift {0};
  while (left != right)
  {
    left >> 1u;
    right >> 1u;
    shift++;
  }

  return left << shift;
}

int BitwiseANDOfNumbersRange::common_mask(int left, int right)
{
  // In bits, this is 11..1.
  uint32_t mask {~0u};

  // Stop condition is 0 = 0.
  while ((left & mask) != (right & mask))
  {
    mask << 1u;
  }

  return (left & mask);
}

//------------------------------------------------------------------------------
/// 215. Kth Largest Element in an Array
//------------------------------------------------------------------------------

int KthLargestElementInAnArray::brute_force(vector<int>& nums, int k)
{
  const int N {static_cast<int>(nums.size())};

  // Practice with quick sort.

  std::function<void(vector<int>&, const int, const int)> quick_sort =
    [&](vector<int>& nums, const int low, const int high) -> void
  {
    // low >= high is a stopping condition.
    if (low < high)
    {
      // Choose, arbitrarily, the "last" element as the "pivot" value to compare
      // other values against.
      const int pivot {nums[high]};

      // Initialize to be less than low to prepare it to track the next index
      // (position) to place lesser values in; low - 1 represents that we hadn't
      // found a lesser value to the pivot at all.
      int i {low - 1};

      for (int j {low}; j < high; ++j)
      {
        // Move all values less than pivot to the left of it.
        if (nums[j] < pivot)
        {
          ++i;
          swap(nums[j], nums[i]);
        }
      }

      // Move the pivot to the end of the lesser values. We're sure that the
      // values on the right are larger than the pivot.
      swap(nums[i + 1], nums[high]);

      const int pivot_index {i + 1};

      // The pivot element is in the correct, sorted position, as all elements
      // to the left of it is smaller and all elements to the right is larger.
      quick_sort(nums, low, pivot_index - 1);
      quick_sort(nums, pivot_index + 1, high);
    }
  };

  quick_sort(nums, 0, nums.size() - 1);

  return nums[N - k];
}

int KthLargestElementInAnArray::find_kth_largest(vector<int>& nums, int k)
{
  const int N {static_cast<int>(nums.size())};

  priority_queue<int> largest {};

  for (const auto num : nums)
  {
    largest.push(num);
  }

  for (int i {0}; i < k - 1; ++i)
  {
    largest.pop();
  }

  return largest.top();
}

//------------------------------------------------------------------------------
/// 230. Kth Smallest Element in a BST
/// Key idea: by the properties of the BST, by induction, we expect that the
/// smallest element is all the way to the left.
//------------------------------------------------------------------------------
int KthSmallestElementInABST::kth_smallest_iterative(TreeNode* root, int k)
{
  TreeNode* current_ptr {root};
  stack<TreeNode*> unvisited_nodes {};
  int count {k};

  while (current_ptr != nullptr || !unvisited_nodes.empty())
  {
    // From the current node, traverse all the left-most nodes because it will
    // be the smallest. We'll not only do this from the root, but also any
    // "right" side child we encounter because we want the kth smallest.
    while (current_ptr != nullptr)
    {
      unvisited_nodes.push(current_ptr);
      current_ptr = current_ptr->left_;
    }

    current_ptr = unvisited_nodes.top();
    unvisited_nodes.pop();

    --count;
    if (count == 0)
    {
      return current_ptr->value_;
    }

    current_ptr = current_ptr->right_;
  }

  // We only expect to reach this if the length of the tree is less than k.
  return -1;
}

int KthSmallestElementInABST::kth_smallest_recursive(TreeNode* root, int k)
{
  int count {k};
  int result {-1};

  function<void(TreeNode*, int&, int&)> step = [&](
    TreeNode* node,
    int& count,
    int& result)
  {
    // We reached the ends of a traversal.
    if (node == nullptr)
    {
      return;
    }

    step(node->left_, count, result);

    --count;
    if (count == 0)
    {
      result = node->value_;
    }

    step(node->right_, count, result);
  };

  step(root, count, result);

  return result;
}

//------------------------------------------------------------------------------
/// 235. Lowest Common Ancestor of a Binary Search Tree
//------------------------------------------------------------------------------
TreeNode*
  LowestCommonAncestorOfABinarySearchTree::lowest_common_ancestor_recursive(
  TreeNode* root,
  TreeNode* p,
  TreeNode* q)
{
  function<TreeNode*(TreeNode*, TreeNode*, TreeNode*)> step = [&](
    TreeNode* node,
    TreeNode* p,
    TreeNode* q)
  {
    if (node == nullptr)
    {
      return node;
    }

    const int current_value {node->value_};

    if (current_value == p->value_ || current_value == q->value_)
    {
      return node;
    }

    // Any common ancestor would be on the left of the current node.
    if (current_value > p->value_ && current_value > q->value_)
    {
      return step(node->left_, p, q);
    }

    // Any common ancestor would be on the right of the current node.
    if (current_value < p->value_ && current_value < q->value_)
    {
      return step(node->right_, p, q);
    }

    // Logically, p and q are on opposite sides of the current node. Thus, the
    // common ancestor would be this node.
    return node;
  };

  return step(root, p, q);
}

TreeNode*
  LowestCommonAncestorOfABinarySearchTree::lowest_common_ancestor_iterative(
  TreeNode* root,
  TreeNode* p,
  TreeNode* q)
{
  TreeNode* node {root};

  while (node != nullptr)
  {
    const int current_value {node->value_};

    if (current_value == p->value_ || current_value == q->value_)
    {
      return node;
    }

    // Any common ancestor would be on the left of the current node.
    if (current_value > p->value_ && current_value > q->value_)
    {
      node = node->left_;
      continue;
    }

    // Any common ancestor would be on the right of the current node.
    if (current_value < p->value_ && current_value < q->value_)
    {
      node = node->right_;
      continue;
    }

    // Logically, p and q are on opposite sides of the current node. Thus, the
    // common ancestor would be this node.
    return node;
  }

  // We expect to be reached if node is a nullptr, indicating no common ancestor
  // to both p and q (for example, root is a child to both p and q).
  return node;
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
/// 260. Single Number III
/// https://leetcode.com/problems/single-number-iii/description/
/// Recall XOR (^) operator and its group properties,
/// self-inverse, associativity, commutativity, 0 identity.
/// e.g. 1,1,3,5. 3=11, 5=101 (bit values, respectively). 3^5=(110)
/// Once you have 
//------------------------------------------------------------------------------
vector<int> SingleNumberIII::single_number(vector<int>& nums)
{
  static constexpr int NUMBER_OF_BITS {32};
  array<int, NUMBER_OF_BITS> bit_count {};

  return {};
}

//------------------------------------------------------------------------------
/// 271. String Encode and Decode
//------------------------------------------------------------------------------
string StringEncodeAndDecode::encode(vector<string>& strs)
{
  stringstream header_stream {};

  // Given strs.length < 100,
  // Dedicate 2 digits to the total size of strs; pad by 0 for a single digit
  // size.
  header_stream << std::setw(2) << std::setfill('0') << strs.size();

  string data {};

  // Because 1 byte = 2^8 (8 bits) = 256, we can represent the length of each
  // str in strs by 1 byte. Consider 1 byte * 100 = 100 bytes. A hexadecimal
  // number of 2 digits represents 1 byte.
  for (const string& str : strs)
  {
    header_stream << std::hex << std::setw(2) << std::setfill('0') <<
      str.size();

    data += str;
  }

  return header_stream.str() + data;
}

vector<string> StringEncodeAndDecode::decode(string s)
{
  stringstream ss {};

  // Decode first 2 chars to be the total length of the resulting list of
  // strings, N.
  ss << s.substr(0, 2);
  size_t N {};
  ss >> N;

  // Given the total length N, we expect N 2-digit hexadecimal values until the
  // data payload.
  // Pointer to the start of the header after the total size N.
  const char* header_sizes {s.c_str() + 2};
  const char* data {s.c_str() + 2 + N * 2};
  vector<string> extracted_strings {};

  for (size_t i {0}; i < N; ++i)
  {
    // Extract 2 characters (hexadecimal size)
    // Get the next 2 chars.
    string hex_size_str(header_sizes, 2);

    // Move pointer forward by 2.
    header_sizes += 2;

    // You NEED to create a new stringstream so to effectively clear the
    // previous stringstream or simply create a new stringstream.
    stringstream hex_ss {};
    size_t hex_size {};
    hex_ss << std::hex << hex_size_str;
    hex_ss >> hex_size;

    string str(data, hex_size);

    extracted_strings.push_back(str);

    data += hex_size;
  }

  return extracted_strings;
}

string StringEncodeAndDecode::encode_with_prefix_neet(vector<string>& strs)
{
  string encoded {};

  stringstream header_stream {};

  for (const string& str : strs)
  {
    header_stream << std::hex << std::setw(2) << std::setfill('0')
      << str.size();

    encoded += header_stream.str();
    encoded += "#";
    encoded += str;

    // Clear any error flags (e.g. EOF flags).
    header_stream.clear();
    // Reset content of stringstream.
    header_stream.str("");
  }

  return encoded;
}

vector<string> StringEncodeAndDecode::decode_with_prefix_neet(string s)
{
  // Consider using 2 iterations to check if we've gotten to the end of the
  // string s. Maybe during transmission, the string s was corrupted so the
  // sizes of strings aren't exact.

  const size_t N {s.size()};

  const char* ptr {s.c_str()};

  size_t i {0};

  stringstream header_stream {};

  vector<string> extracted_strings {};

  while (*ptr != '\0' && i < N)
  {
    string str_hex_size(ptr, 2);

    header_stream << std::hex << str_hex_size;
    size_t str_size {};
    header_stream >> str_size;

    i += 3;
    ptr += 3;

    extracted_strings.push_back(string(ptr, str_size));

    i += str_size;
    ptr += str_size;

    header_stream.clear();
    header_stream.str("");
  }

  return extracted_strings;
}

//------------------------------------------------------------------------------
/// 289. Game of Life
//------------------------------------------------------------------------------

void GameOfLife::game_of_life(vector<vector<int>>& board)
{
  const int M {static_cast<int>(board.size())};
  const int N {static_cast<int>(board[0].size())};
  using Indices = tuple<int, int>;
  auto is_valid_indices = [M, N](const Indices& indices) -> bool
  {
    const auto [i, j] = indices;
    return (i >= 0 && i < M && j >= 0 && j < N);
  };

  static constexpr array<Indices, 8> directions {
    // right
    make_tuple(0, 1),
    // left
    make_tuple(0, -1),
    // down
    make_tuple(1, 0),
    // up
    make_tuple(-1, 0),
    // down-right
    make_tuple(1, 1),
    // down-left
    make_tuple(1, -1),
    // up-right
    make_tuple(-1, 1),
    // down-left
    make_tuple(-1, -1)};

  static constexpr int ALIVE {1};
  static constexpr int DEAD {0};
  static constexpr int ALIVE_GOING_TO_DEAD {2};
  static constexpr int DEAD_GOING_TO_ALIVE {-1};

  auto get_next_state = [M, N, &is_valid_indices, &board](
    const int i,
    const int j) -> int
  {
    int live_neighbors {0};

    for (const auto& [di, dj] : directions)
    {
      const Indices new_indices {i + di, j + dj};
      if (is_valid_indices(new_indices))
      {
        const int new_value {board[get<0>(new_indices)][get<1>(new_indices)]};
        if (new_value == ALIVE || new_value == ALIVE_GOING_TO_DEAD)
        {
          live_neighbors++;
        }
      }
    }

    const int current_value {board[i][j]};

    // If under-population or over-population, die.
    if (live_neighbors < 2 || live_neighbors > 3)
    {
      if (current_value == ALIVE)
      {
        return ALIVE_GOING_TO_DEAD;
      }
      return current_value;
    }

    // If reproduction, come to life, else live on.
    if (live_neighbors == 3)
    {
      if (current_value == DEAD)
      {
        return DEAD_GOING_TO_ALIVE;
      }
      return current_value;
    }
    // live_neighbors == 2 expected, and if alive, it lives on. If dead, it
    // stays dead.
    return current_value;
  };

  for (int i {0}; i < M; ++i)
  {
    for (int j {0}; j < N; ++j)
    {
      board[i][j] = get_next_state(i, j);
    }
  }

  for (int i {0}; i < M; ++i)
  {
    for (int j {0}; j < N; ++j)
    {
      if (board[i][j] == ALIVE_GOING_TO_DEAD)
      {
        board[i][j] = DEAD;
      }

      if (board[i][j] == DEAD_GOING_TO_ALIVE)
      {
        board[i][j] = ALIVE;
      }
    }
  }
}

//------------------------------------------------------------------------------
/// 347. Top K Frequent Elements
//------------------------------------------------------------------------------

vector<int> TopKFrequentElements::brute_force(vector<int>& nums, int k)
{
  unordered_map<int, int> number_to_count {};

  for (const int num : nums)
  {
    number_to_count[num]++;
  }

  vector<pair<int, int>> number_to_counts {};

  for (const auto [number, count] : number_to_count)
  {
    number_to_counts.push_back({number, count});
  }

  sort(
    number_to_counts.begin(),
    number_to_counts.end(),
    [](const auto pair1, const auto pair2)
    {
      return pair1.second > pair2.second;
    });

  vector<int> results {};

  for (int l {0}; l < k; ++l)
  {
    if (l < number_to_counts.size())
    {
      results.push_back(number_to_counts[l].first);
    }
  }

  return results;
}

// O(N) time complexity.
// https://youtu.be/YPTqKIgVk-k?si=KbHT9oTwGare3dkl
vector<int> TopKFrequentElements::bucket_sort(vector<int>& nums, int k)
{
  // For each num in nums, use hash map (unordered_map) to map num to count
  // (frequency).
  unordered_map<int, int> num_to_count {};
  // O(N) time complexity
  for (const int num : nums)
  {
    num_to_count[num]++;
  }

  // Because nums.length <= 10^5, for bucket sort to work it needs a count to
  // std::vector<int> collection of num's with that count.
  // Be careful of total number of indices where each index is count. Since a
  // single number can show up nums.size() times, count=nums.size() must be
  // accounted for.
  // index=0...(nums.size()).
  // i.e.
  // for count_to_nums, index for count_to_nums, index = frequency count, so if
  // i = 3, frequency or count is 3.
  // count_to_nums[3] is a list of all num's (elements?) with frequency or count
  // 3.
  vector<vector<int>> count_to_nums (nums.size() + 1, vector<int>{});

  // O(N) time complexity.
  for (const auto [num, count] : num_to_count)
  {
    count_to_nums[count].push_back(num);
  }

  // Because the array is inherently sorted by its index, which represents count
  // in this case, access array until k most frequent elements obtained.
  int lth_most_frequent {k};
  int count_index {static_cast<int>(nums.size())};

  vector<int> result {};

  // O(N) time complexity. We may need to go through entire list of buckets to
  // collect k elements.
  while (lth_most_frequent > 0 && count_index >= 0)
  {
    for (const int num : count_to_nums[count_index])
    {
      result.push_back(num);
      lth_most_frequent--;
      if (lth_most_frequent == 0)
      {
        break;
      }
    }
    count_index--;
  }

  return result;
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
/// 371. Sum of Two Integers
/// XOR, since 0^0=1^1=0, and 0^1=1^0=1, XOR immediately gives the bits of the
/// resulting sum, but not any carry.
/// Key insight: & (and) finds where carry bits should be generated; only 1&1=1.
//------------------------------------------------------------------------------

int SumOfTwoIntegers::get_sum(int a, int b)
{
  // Example 2 + 3 = 5. 1,0 + 1,1 \equiv 10 + 11.
  // carry: 10. sum: 01. carry <<= 1 = 100. sum ^= carry = 101.

  // Has a 1 bit where the next "left" or next higher bit should have a carry.
  int carry {a & b};

  carry <<= 1;

  // Has bit values if there was no carry.
  int sum {a ^ b};

  // If an integer is overly negative, the carry can be negative as well, not
  // necessarily strictly positive.
  while (carry != 0)
  {
    const int previous_sum {sum};
    sum ^= carry;
    carry &= previous_sum;
    carry <<= 1;
  }

  return sum;
}


//------------------------------------------------------------------------------
/// 378. Kth Smallest Element in a Sorted Matrix
//------------------------------------------------------------------------------

int KthSmallestElementInASortedMatrix::brute_force(
  vector<vector<int>>& matrix,
  int k)
{
  const int N {static_cast<int>(matrix.size())};
  vector<int> sorted {};

  for (int i {0}; i < N; ++i)
  {
    for (int j {0}; j < N; ++j)
    {
      sorted.emplace_back(matrix[i][j]);
    }
  }

  sort(sorted.begin(), sorted.end());

  return sorted[k - 1];
}

int KthSmallestElementInASortedMatrix::kth_smallest(
  vector<vector<int>>& matrix,
  int k)
{
  const int N {static_cast<int>(matrix.size())};

  // std::priority_queue expected to default to a max heap and that it auto-
  // magically does max heapify, swapping each new insert with its parent
  // until heap condition is restored (all 2 children of each parent is of
  // lesser value to parent's value).
  priority_queue<int> max_heap {};

  int start {0};

  for (int i {0}; i < N; ++i)
  {
    for (int j {0}; j < N; ++j)
    {
      max_heap.push(matrix[i][j]);
    }
  }

  // The max heap will be of size k and the smallest value will be the leaf -
  // the top will be the kth smallest.
  while (max_heap.size() > k)
  {
    max_heap.pop();
  }

  return max_heap.top();
}

//------------------------------------------------------------------------------
/// 424. Longest Repeating Character Replacement
/// https://leetcode.com/problems/longest-repeating-character-replacement/description/
/// Key idea: Sliding window technique.
//------------------------------------------------------------------------------
int LongestRepeatingCharacterReplacement::character_replacement(string s, int k)
{
  const int N {static_cast<int>(s.size())};

  if (k == N)
  {
    return N;
  }

  // Since we're limited to uppercase letters, use an array.
  array<int, 26> character_value_to_counts {};

  // Use this index to shrink the window.
  int start {0};

  // Count of the most frequent character seen in the current window.
  int max_count {0};

  int length {0};

  // Expand the window until the very last element.
  for (int j {0}; j < N; ++j)
  {
    // Account for expanding the window and seeing a new character.
    const char current_character {s[j]};
    const int current_character_index {
      static_cast<int>(current_character - 'A')};
    character_value_to_counts[current_character_index]++;

    // The window should be expanded if the number of replacements, k, plus the
    // count of the most frequent character in window is greater than or equal
    // to window size. If not, shrink window from start.

    max_count = max(
      max_count,
      character_value_to_counts[current_character_index]);

    length = max(
      length,
      min(max_count + k, j - start + 1));

    if (j - start + 1 > (max_count + k))
    {
      character_value_to_counts[static_cast<int>(s[start] - 'A')]--;

      start++;
    }
  }

  return length;
}

//------------------------------------------------------------------------------
/// 435. Non-overlapping intervals
/// Key idea: sort array.
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
/// 438. Find all anagrams in a string
//------------------------------------------------------------------------------
vector<int> FindAllAnagramsInAString::find_anagrams(string s, string p)
{
  const int P {static_cast<int>(p.size())};
  const int S {static_cast<int>(s.size())};
  if (S < P)
  {
    return {};
  }

  // Use array for O(1) access. We already are given that the letters are only
  // lowercase letters.
  array<int, 26> sletter_to_counts {};
  array<int, 26> pletter_to_counts {};

  // O(|p|) time complexity.
  for (const char c : p)
  {
    pletter_to_counts[static_cast<int>(c - 'a')] += 1;
  }

  int start {0};
  vector<int> start_indicies {};

  for (int i {0}; i < S; ++i)
  {
    const int current_c {static_cast<int>(s[i] - 'a')};

    sletter_to_counts[current_c] += 1;

    if (i - start == (P - 1))
    {
      bool is_match {true};

      for (int i {0}; i < 26; ++i)
      {
        if (sletter_to_counts[i] != pletter_to_counts[i])
        {
          is_match = false;
        }
      }

      if (is_match)
      {
        start_indicies.emplace_back(start);
      }

      sletter_to_counts[static_cast<int>(s[start] - 'a')] -= 1;
      ++start;
    }
  }
  return start_indicies;
}

//------------------------------------------------------------------------------
/// 516. Longest Palindromic Subsequence
//------------------------------------------------------------------------------
int LongestPalindromicSubsequence::longest_palindrome_subsequence(string s)
{
  const int N {static_cast<int>(s.size())};
  // O(N^2) space complexity.
  // 0 <= i, j <= N - 1
  // Gives the maximum length of a palindromic substring between indices i, j
  // and is inclusive.
  vector<vector<int>> max_lengths (N, vector<int>(N, false));

  // Every single character is a palindrome.
  for (int i {0}; i < N; ++i)
  {
    max_lengths[i][i] = 1;
  }

  // check for every substring length.
  for (int length {2}; length <= N; ++length)
  {
    // Check, for each finite length, each starting point.
    for (int i {0}; i <= N - length; ++i)
    {
      int j {i + length - 1};

      if (s[i] == s[j])
      {
        max_lengths[i][j] = 2 + (length == 2 ? 0 : max_lengths[i + 1][j - 1]);
      }
      else
      {
        // We take care of the case where we can "skip" over letters that don't
        // make a palindrome by getting the "previous" maximum.
        // Notice we take the max of either i + 1 or j - 1 because either s[i]
        // or s[j] contributed to not making a palindrome.
        max_lengths[i][j] = max(max_lengths[i + 1][j], max_lengths[i][j - 1]);
      }
    }
  }

  return max_lengths[0][N - 1];
}

//------------------------------------------------------------------------------
/// \name 542. 01 Matrix
/// \url https://leetcode.com/problems/01-matrix/
//------------------------------------------------------------------------------
vector<vector<int>> Update01Matrix::update_matrix(vector<vector<int>>& mat)
{
  static const vector<array<int, 2>> directions {
    {1, 0},
    {0, 1},
    {-1, 0},
    {0, -1}};

  const int M {static_cast<int>(mat.size())};
  const int N {static_cast<int>(mat[0].size())};

  vector<vector<int>> distances (M, vector<int>(N, INT_MAX));

  auto is_in_bound = [M, N](const int i, const int j)
  {
    return (i >= 0) && (j >= 0) && (i < M) && (j < N);
  };

  // Add indices of cells with 0.
  queue<pair<int, int>> q {};

  for (int i {0}; i < M; ++i)
  {
    for (int j {0}; j < N; ++j)
    {
      if (mat[i][j] == 0)
      {
        q.push({i, j});
        distances[i][j] = 0;
      }
    }
  }

  while (!q.empty())
  {
    auto [i, j] = q.front();
    q.pop();

    for (const auto& [di, dj] : directions)
    {
      int new_i {i + di};
      int new_j {j + dj};

      if (is_in_bound(new_i, new_j))
      {
        if (distances[new_i][new_j] > distances[i][j] + 1)
        {
          distances[new_i][new_j] = distances[i][j] + 1;
          q.push({new_i, new_j});
        }
      }
    }
  }

  return distances;


      /*
      const int current_value {mat[i][j]};

      if (current_value != 0)
      {
        int minimum_distance {1};

        queue<pair<int, int>> to_visit {};
        to_visit.push(make_pair<int, int>(move(i), move(j)));

        while (!to_visit.empty())
        {
          const auto current_ij = to_visit.front();
          to_visit.pop();

          for (const auto& direction : directions)
          {
            int new_i {current_ij.first + direction[0]};
            int new_j {current_ij.second + direction[1]};

            if (is_in_bound(new_i, new_j) && mat[new_i][new_j] == 0)
            {
              // Empty queue because we no longer have to visit cells.
              while (!to_visit.empty())
              {
                to_visit.pop();
              }
              break;
            }
            else if (is_in_bound(new_i, new_j))
            {
              to_visit.push(make_pair<int, int>(move(new_i), move(new_j)));
            }
          }

          if (!to_visit.empty())
          {
            minimum_distance += 1;
          }
        }
        mat[i][j] = minimum_distance;
      }
    }
  }

  return mat;
      */
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

//------------------------------------------------------------------------------
/// \name 567. Permutation in String
/// https://leetcode.com/problems/permutation-in-string/
/// Given two strings s1 and s2, return true if s2 contains a permutation of s1,
/// or false otherwise.
///
/// In other words, return true if one of s1's permutations is the substring of
/// s2.
/// s1 and s2 consist of lowercase English letters.
/// https://youtu.be/UbyhOgBN834?si=vu09VPh6pdTqUu05
//------------------------------------------------------------------------------
bool PermutationInString::check_inclusion(string s1, string s2)
{
  // Use the fact that we only have lowercase English letters.
  array<int, 26> s1_letter_index_to_count {};

  for (const char c : s1)
  {
    s1_letter_index_to_count[static_cast<int>(c-'a')]++;
  }

  array<int, 26> letter_index_to_count {};

  // "left" pointer to the sliding window.
  int start_index {0};

  const int Ns1 {static_cast<int>(s1.size())};

  for (int r {0}; r < static_cast<int>(s2.size()); ++r)
  {
    ++letter_index_to_count[static_cast<int>(s2[r] - 'a')];

    if ((r  - start_index + 1) == Ns1)
    {
      if (letter_index_to_count == s1_letter_index_to_count)
      {
        return true;
      }

      --letter_index_to_count[static_cast<int>(s2[start_index] - 'a')];
      ++start_index;
    }
  }

  return false;

  /*
  unordered_map<char, int> s1_char_to_count {};
  // O(s) time
  for (const char c : s1)
  {
    ++s1_char_to_count[c];
  }

  int start_index {0};

  unordered_map<char, int> char_to_count {s1_char_to_count};

  int start_index {0};
  for (int r {0}; r < static_cast<int>(s2.size()); ++r)
  {
    if (s1_char_to_count.count(s2[r]) == 0)
    {
      start_index = r + 1;      
      char_to_count = s1_char_to_count;
    }
    else
    {
      if (char_to_count[s2[r]] == 0)
      {
        //char_to_count[s2[start_index]] += 1;
        start_index += 1;
      }
      else
      {
        char_to_count[s2[r]]--;

        if (r - start_index + 1 == static_cast<int>(s1.size()))
        {
          for (int i {start_index}; i <= r; ++i)
          {
            if (char_to_count[s2[i]] != 0)
            {
              char_to_count[s2[start_index]] += 1;
              start_index += 1;              
              break;
            }
          }
          return true;
        }
      }
    }
  }

  return false;
  */
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

//------------------------------------------------------------------------------
/// 692. Top K Frequent Words
/// https://leetcode.com/problems/top-k-frequent-words/description/ 
/// Constraints:
///
/// 1 <= words.length <= 500
/// 1 <= words[i].length <= 10
/// words[i] consists of lowercase English letters.
/// k is in the range [1, The number of unique words[i]]
///
/// Follow-up: Could you solve it in O(n log(k)) time and O(n) extra space?
//------------------------------------------------------------------------------
vector<string> TopKFrequentWords::brute_force(vector<string>& words, int k)
{
  // word to count map. O(N) space.
  unordered_map<string, int> word_to_count {};
  // O(N) time complexity
  for (const string& word : words)
  {
    // O(1) time complexity amoritized for insertion, and lookup.
    word_to_count[word]++;
  }

  // O(N) space (all the words get stored again).
  unordered_map<int, vector<string>> count_to_words {};
  // O(N) time.
  for (const auto& [word, count] : word_to_count)
  {
    // O(1) time amoritized find and insert.
    count_to_words[count].push_back(word);
  }

  // Sort each bucket of words by lexicographical order.
  // O(N log N) time, as sort is O(k log k) time for k words to sort.
  for (auto& [_, words] : count_to_words)
  {
    sort(words.begin(), words.end());
  }

  vector<pair<int, vector<string>>> count_to_words_sorted {};
  for (const auto& [count, words] : count_to_words)
  {
    // O(1) insertion from the back of an array.
    count_to_words_sorted.push_back({count, words});
  }

  // O(k log k) time for sorting k elements.
  sort(
    count_to_words_sorted.begin(),
    count_to_words_sorted.end(),
    [](const auto& a, const auto& b) -> bool
    {
      return a.first > b.first;
    });

  vector<string> result {};
  int count_to_words_sorted_index {0};

  // O(k) time
  while (result.size() < k &&
    count_to_words_sorted_index < count_to_words_sorted.size())
  {
    const auto& [count, words] = count_to_words_sorted[
      count_to_words_sorted_index];
    // O(N) time
    for (const string& word : words)
    {
      result.push_back(word);
      if (result.size() == k)
      {
        break;
      }
    }
    count_to_words_sorted_index++;
  }

  return result;
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

//------------------------------------------------------------------------------
/// 1297. Maximum Number of Occurrences of a Substring
/// https://leetcode.com/problems/maximum-number-of-occurrences-of-a-substring/description/
/// Constraints:
///
/// 1 <= s.length <= 105
/// 1 <= maxLetters <= 26
/// 1 <= minSize <= maxSize <= min(26, s.length)
/// s consists of only lowercase English letters.
//------------------------------------------------------------------------------

int MaximumNumberOfOccurrencesOfASubstring::max_freq(
  string s,
  int max_letters,
  int min_size,
  int max_size)
{
  auto is_valid_substring = [&](const string& substring) -> bool
  {
    unordered_set<char> unique_characters {};
    for (const char c : substring)
    {
      unique_characters.insert(c);
    }
    return unique_characters.size() <= max_letters;
  };

  const int N {static_cast<int>(s.size())};

  // space complexity: N * (max_size - min_size + 1) ->
  // O(N * (max_size - min_size + 1))
  unordered_map<string, int> substring_to_count {};

  // time complexity: O(N)
  for (int i {0}; i < (N - min_size + 1); ++i)
  {
    for (int k {min_size}; k <= max_size; ++k)
    {
      // From
      // https://en.cppreference.com/w/cpp/string/basic_string/substr.html
      // if requested substring extends past the end of the string, i.e. the
      // count is greater than size() - pos, the returned substring is
      // [pos, size()], so that we need to check if the substring doesn't extend
      // past the end of the string.
      if ((i + k) <= N)
      {
        const string substring {s.substr(i, k)};
        if (is_valid_substring(substring))
        {
          substring_to_count[substring]++;
        }
      }
    }
  }

  int max_freq {0};
  for (const auto& [_, count] : substring_to_count)
  {
    max_freq = max(max_freq, count);
  }

  return max_freq;
}

//------------------------------------------------------------------------------
/// Observe that the substring with the most occurrences must always be the
/// substring of minimal size.
//------------------------------------------------------------------------------

int MaximumNumberOfOccurrencesOfASubstring::max_freq_with_bitfield(
  string s,
  int max_letters,
  int min_size,
  int max_size)
{
  const int N {static_cast<int>(s.size())};
  unordered_map<string, int> substring_to_count {};

  auto is_valid_substring = [&](const int i) -> bool
  {
    uint32_t bitfield {0};
    int unique_characters_seen {0};
    // Only need to check up to the min size because the substring of most
    // occurrences must be of minimal size.
    for (int j {i}; j < (i + min_size); ++j)
    {
      const int index {static_cast<int>(s[j] - 'a')};
      const bool is_bit_set_high {(bitfield & (1u << index)) != 0};
      if (!is_bit_set_high)
      {
        unique_characters_seen++;
        bitfield |= (1 << index);
      }
    }

    return unique_characters_seen <= max_letters;
  };

  for (int i {0}; i < (N - min_size + 1); ++i)
  {
    if (is_valid_substring(i))
    {
      substring_to_count[s.substr(i, min_size)]++;
    }
  }

  int max_freq {0};
  for (const auto& [_, count] : substring_to_count)
  {
    max_freq = max(max_freq, count);
  }

  return max_freq;
}

int MaximumNumberOfOccurrencesOfASubstring::max_freq_with_sliding_window(
  string s,
  int max_letters,
  int min_size,
  int max_size)
{
  const int N {static_cast<int>(s.size())};
  unordered_map<string, int> substring_to_count {};

  array<int, 26> letter_index_to_count {};
  int unique_characters_seen {0};

  // Initialize first window.
  for (int i {0}; i < min_size; ++i)
  {
    const int index {static_cast<int>(s[i] - 'a')};
    letter_index_to_count[index]++;
    if (letter_index_to_count[index] == 1)
    {
      unique_characters_seen++;
    }
  }

  // Process first window.
  if (unique_characters_seen <= max_letters)
  {
    substring_to_count[s.substr(0, min_size)]++;
  }

  for (int i {min_size}; i < N; ++i)
  {
    const int left_most_index {static_cast<int>(s[i - min_size] - 'a')};
    letter_index_to_count[left_most_index]--;
    if (letter_index_to_count[left_most_index] == 0)
    {
      unique_characters_seen--;
    }

    const int right_most_index {static_cast<int>(s[i] - 'a')};
    letter_index_to_count[right_most_index]++;
    if (letter_index_to_count[right_most_index] == 1)
    {
      unique_characters_seen++;
    }

    if (unique_characters_seen <= max_letters)
    {
      substring_to_count[s.substr(i - min_size + 1, min_size)]++;
    }
  }

  int max_freq {0};
  for (const auto& [_, count] : substring_to_count)
  {
    max_freq = max(max_freq, count);
  }

  return max_freq;
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
