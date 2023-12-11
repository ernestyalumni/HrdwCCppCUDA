#include "MediumProblems.h"

#include <array>
#include <cstddef> // std::size_t
#include <limits>
#include <stack>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include <iostream>

using std::array;
using std::size_t;
using std::stack;
using std::string;
using std::unordered_set;
using std::vector;

namespace Algorithms
{
namespace LeetCode
{

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
