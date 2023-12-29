#include "HardProblems.h"

#include <algorithm> // std::swap
#include <array>
#include <climits>
#include <deque>
#include <limits.h> // INT_MAX, INT_MIN
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

using std::array;
using std::deque;
using std::min;
using std::stack;
using std::string;
using std::swap;
using std::unordered_map;
using std::vector;

namespace Algorithms
{
namespace LeetCode
{

//------------------------------------------------------------------------------
/// 41. First Missing Positive
//------------------------------------------------------------------------------
int FirstMissingPositive::first_missing_positive(vector<int>& nums)
{
  // O(N) time complexity if it's not recorded.
  const int N {static_cast<int>(nums.size())};

  for (int i {0}; i < N; ++i)
  {
    while (nums[i] > 0 && nums[i] <= N &&
      // If nums[i] was in its sorted, "correct", position, then nums[i] ==
      // i + 1. So then [nums[i] - 1] = [i + 1 - 1] = [i] and so
      // nums[i] == nums[i].
      (nums[nums[i] - 1] != nums[i]))
    {
      swap(nums[i], nums[nums[i] - 1]);

      // This is guaranteed to stop because each time we swap the value to its
      // sorted, "correct", position and there are only finite number of nums.
    }
  }

  for (int i {0}; i < N; ++i)
  {
    if (nums[i] != i + 1)
    {
      return i + 1;
    }
  }

  return N + 1;
}

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

//------------------------------------------------------------------------------
/// 239. Sliding Window Maximum
//------------------------------------------------------------------------------
vector<int> SlidingWindowMaximum::max_sliding_window(vector<int>& nums, int k)
{
  const int N {static_cast<int>(nums.size())};

  if (k == 1)
  {
    return nums;
  }
  else if (k == N)
  {
    int maximum_value {INT_MIN};

    for (int i {0}; i < N; ++i)
    {
      if (nums[i] > maximum_value)
      {
        maximum_value = nums[i];
      }
    }

    return {maximum_value};
  }

  vector<int> maxima {};
  // Keep the maximum at top by checking for each current element with top().
  // Place the others at the back, checking the back.
  // Remember, this keeps the indices of the elements, not the values.
  deque<int> sorted_window {};

  // From i == 0, fill up the deque until the first k-sized window is filled up.
  for (int i {0}; i < N; ++i)
  {
    const int current_value {nums[i]};

    // Check if our maximum has moved out of the k-sized window. We only care
    // for the maximum being within the window.
    while (!sorted_window.empty() && sorted_window.front() <= i - k)
    {
      sorted_window.pop_front();
    }

    // Ensure deque is in descending order by removing any elements from the
    // back that are smaller or equal to the current element. This ensures the
    // deque is in descending order of values. Any smaller element in the back
    // will never be the maximum when the current element in the window is both
    // larger and more recently added.
    while (!sorted_window.empty() &&
      nums[sorted_window.back()] <= current_value)
    {
      sorted_window.pop_back();
    }

    sorted_window.push_back(i);

    // This is also ok.
    /*
    while (!sorted_window.empty() && sorted_window.front() <= i - k)
    {
      sorted_window.pop_front();
    }
    */
    // This is also ok.
    /*
    if (!sorted_window.empty() && sorted_window.front() <= i - k)
    {
      sorted_window.pop_front();
    }
    */

    // Start adding the maximum once we've seen the first k-sized window.
    if (i >= k - 1)
    {
      maxima.push_back(nums[sorted_window.front()]);
    }
  }

  return maxima;
}
  /* Before when I didn't know to use a deque.
  const int K {min(k, N - k)};

  // First element shall always have the maximum value and absolute index.
  // Second element will have second largest value, and so on.
  vector<int> sorted_window (K, INT_MIN);

  for (int i {0}; i < N - k; ++i)
  {
    if (i == 0)
    {

    }
  }
  */

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

/// 1547. Minimum Cost to Cut a Stick.

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

//------------------------------------------------------------------------------
/// 1944. Number of Visible People in a Queue
//------------------------------------------------------------------------------

vector<int> NumberOfVisiblePeopleInAQueue::can_see_persons_count(
  vector<int>& heights)
{
  const int N {static_cast<int>(heights.size())};
  // Tracks the index of each person upon which we still need to consider their
  // height.
  // Use a stack for LIFO ordering: we want to check for each current person
  // the heights of each person on right from the next adjacent right, until
  // the end.
  stack<int> unresolved_people {};

  vector<int> visible_persons (N, 0);

  // Key insight was to iterate from the right. It wasn't obvious to me.
  for (int i {N - 1}; i >= 0; --i)
  {
    const int current_height {heights[i]};

    while (!unresolved_people.empty() &&
      // The current person can see above the previous person on the right.
      (current_height > heights[unresolved_people.top()]))
    {
      visible_persons[i]++;
      // Since the current person is taller than the previous person on the
      // right, this person will overshadow them. So for the next person to
      // consider on the left, we can pop this person.
      unresolved_people.pop();
    }

    // Assert that current_height <= heights[unresolved_people.top()], i.e.
    // there's someone taller from the right that the current person can't see
    // past.
    // Increment the number of visible persons because the current person can
    // the taller person.
    if (!unresolved_people.empty())
    {
      visible_persons[i]++;
    }

    unresolved_people.push(i);
  }

  return visible_persons;
}

} // namespace LeetCode
} // namespace Algorithms
