#include "MediumProblems.h"

#include "DataStructures/BinaryTrees.h"

#include <algorithm> // std::sort, std::swap;
#include <array>
#include <climits> // INT_MIN
#include <cstddef> // std::size_t
#include <functional>
#include <limits.h>
#include <limits>
#include <map>
#include <numeric> // std::accumulate
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using DataStructures::BinaryTrees::TreeNode;
using std::array;
using std::function;
using std::max;
using std::min;
using std::priority_queue;
using std::queue;
using std::size_t;
using std::stack;
using std::string;
using std::swap;
using std::unordered_map;
using std::unordered_set;
using std::vector;

namespace Algorithms
{
namespace LeetCode
{

//------------------------------------------------------------------------------
/// 3. Longest Substring Without Repeating Characters
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
      bool is_row_seen {false};

      if (matrix[i][j] == 0)
      {
        if (!is_row_seen)
        {
          is_row_zeroes[i / number_of_bits] |=
            (static_cast<uint64_t>(1) << (i % number_of_bits));
          is_row_seen = true;
        }

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
