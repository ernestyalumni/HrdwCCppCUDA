#include "HardProblems.h"

#include <algorithm>
#include <climits>
#include <unordered_map>
#include <string>
#include <vector>

using std::string;
using std::unordered_map;
using std::vector;

namespace Algorithms
{
namespace LeetCode
{

/// \name 76. Minimum Window Substring
string MinimumWindowSubstring::minimum_window(string s, string t)
{
  const int M {static_cast<int>(s.size())};
  const int N {static_cast<int>(t.size())};

  // Use unordered_map, implemented as a hash table, over map, implemented as a 
  // red-black tree, for O(1) amoritized access.

  // Keep count of each character's frequency in 't'. This helps to know how
  // many of each character we need to find in 's'.
  unordered_map<char, int> t_letter_to_frequencies {};
  // Keep count of how many characters of t as they appear in the current
  // window.
  unordered_map<char, int> t_letter_to_count_in_window {};

  for (const char c : t)
  {
    t_letter_to_frequencies[c]++;
  }

  // Track number of unique_characters seen from t.
  int number_of_unique_characters_seen {0};
 
  const int number_of_unique_t_characters {
    static_cast<int>(t_letter_to_frequencies.size())};
  int start {0};
  int end {0};
  int length {INT_MAX};
  int min_start {0};

  // We expand the window until we acheive the condition.
  // O(M) time complexity.
  while (end < M)
  {
    const char c {s[end]};

    if (t_letter_to_frequencies.count(c) != 0)
    {
      t_letter_to_count_in_window[c]++;

      if (t_letter_to_count_in_window[c] == t_letter_to_frequencies[c])
      {
        number_of_unique_characters_seen += 1;
      }
    }

    // We check if we had obtained the condition: every character in t is
    // included in the window.
    while (
      (start <= end) &&
      (number_of_unique_characters_seen == number_of_unique_t_characters))
    {
      if (end - start + 1 < length)
      {
        length = end - start + 1;
        min_start = start;
      }

      const char c2 {s[start]};

      if (t_letter_to_frequencies.count(c2) != 0)
      {
        t_letter_to_count_in_window[c2]--;

        if (t_letter_to_count_in_window[c2] < t_letter_to_frequencies[c2])
        {
          number_of_unique_characters_seen--;
        }
      }

      start++;
    }

    end++;
  }

  return length == INT_MAX ? "" : s.substr(min_start, length);
}

int ShortestPathInGridWithObstacles::shortest_path(
  vector<vector<int>>& grid,
  int k)
{
  return grid.size() + k; 
}

int WaysToEarnPoints::ways_to_reach_target(
  int target,
  vector<vector<int>>& types)
{
  /*
  std::sort(
    types.begin(),
    types.end(),
    [](const vector<int>& a, const vector<int>&b)
    {
      // https://en.cppreference.com/w/cpp/algorithm/sort
      // Elements are compared using operator<.
      return a[1] < b[1];
    });
  */

  // Each index is the desired target value, and the value at an index is the
  // number of ways to obtain that target value. We just want all possible
  // target values up to a given target.
  vector<int> number_of_ways (target + 1, 0);
  // This is the base case for combinations, "there's exactly one way to achieve
  // 0 by not selecting any questions", and
  // it's the starting point of accumulation, by having this as the starting
  // point for the recursive relationship.
  number_of_ways[0] = 1;

  // Key insight is the necessity to track the number of questions left.
  /*
  vector<int> number_of_questions_left (types.size());
  std::transform(
    types.begin(),
    types.end(),
    number_of_questions_left.begin(),
    [](const std::vector<int>& types_element)
  {
    return types_element[0];
  })
  */

  // O(N) time.
  for (const auto& count_and_mark : types)
  {
    // O(T)
    for (int j {target}; j >= count_and_mark[1]; --j)
    {
      // O(C)
      for (int k {1}; k <= count_and_mark[0] && k * count_and_mark[1] <= j; ++k)
      {
        number_of_ways[j] = (
          number_of_ways[j] +
            number_of_ways[j - k * count_and_mark[1]]) % 1'000'000'007;
      }
    }
  }

  return number_of_ways[target];
}

int MinimumCostToCutStick::minimum_cost_to_cut_stick(int n, vector<int>& cuts)
{
  const int N {static_cast<int>(cuts.size())};

  vector<int> cuts_with_boundaries (N + 2, 0);

  // O(N) time.
  std::copy(cuts.begin(), cuts.end(), cuts_with_boundaries.begin() + 1);

  cuts_with_boundaries[N + 1] = n;

  // O(N log N) time.
  std::sort(cuts_with_boundaries.begin(), cuts_with_boundaries.end());

  // Minimum cost of making all cuts between cuts[i] and cuts[j].
  std::vector<std::vector<int>> minimum_cost (
    N + 2,
    std::vector<int>(N + 2, INT_MAX));

  for (int i {0}; i < N + 1; ++i)
  {
    minimum_cost[i][i] = 0;
    // Set the cost of adjacent cuts to 0.
    minimum_cost[i][i + 1] = 0;
  }

  // Choose a substick to cut via the index to cuts because cuts will have where
  // the stick will ultimately be broken up to.
  for (int l {2}; l <= N + 1; ++l)
  {
    // Choose a starting point amonst the cuts.
    for (int i {0}; i <= N + 1 - l; ++i)
    {
      int j {i + l};
      // Cut points within a sub-stick
      for (int k {i + 1}; k < j; ++k)
      {
        if (minimum_cost[i][k] != INT_MAX && minimum_cost[k][j] != INT_MAX)
        {
          minimum_cost[i][j] = std::min(
            minimum_cost[i][j],
            // (total cost of cut at k, being the total length of substick) +
            // previous minimum costs.
            cuts_with_boundaries[j] -
              cuts_with_boundaries[i] +
              minimum_cost[i][k] +
              minimum_cost[k][j]);
        }
      }
    }
  }

  return minimum_cost[0][N + 1];
}

} // namespace LeetCode
} // namespace Algorithms
