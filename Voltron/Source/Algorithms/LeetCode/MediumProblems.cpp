#include "MediumProblems.h"

#include <cstddef> // std::size_t
#include <string>
#include <tuple>
#include <vector>

#include <iostream>

using std::size_t;
using std::string;
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

} // namespace LeetCode
} // namespace Algorithms
