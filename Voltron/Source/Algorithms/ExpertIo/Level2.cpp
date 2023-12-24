#include "Level2.h"

#include <algorithm> // std::sort, std::swap
#include <cassert>
#include <vector>

using std::vector;

namespace Algorithms
{
namespace ExpertIo
{

namespace BestSeat
{

int best_seat(vector<int> seats)
{
  const int N {static_cast<int>(seats.size())};
  int start {0};
  int end {0};

  int max_start {0};
  int length {-1};
  bool is_in_empty_seats {false};

  while (end < N)
  {
    if (seats[end] == 1)
    {
      if (is_in_empty_seats)
      {
        is_in_empty_seats = false;

        if (((end - start) > 1) && (end - start) > length)
        {
          length = end - start;
          max_start = start;
        }
      }
      
      start = end;
    }
    else
    {
      if (!is_in_empty_seats)
      {
        is_in_empty_seats = true;
      }
    }
    ++end;
  }

  return length == -1 ? length : max_start + length / 2;
}

} // namespace BestSeat

namespace MergeOverlappingIntervals
{

//------------------------------------------------------------------------------
/// \url https://www.algoexpert.io/questions/Merge%20Overlapping%20Intervals
/// \details Hint 1: The problem asks you to merge overlapping intervals. How
/// can you determine if two intervals are overlapping?
//------------------------------------------------------------------------------

bool is_not_overlapping(
  const vector<int>& interval1,
  const vector<int>& interval2)
{
  return (interval1[1] < interval2[0]) &&  (interval2[1] < interval1[0]);
}

//------------------------------------------------------------------------------
/// \details Hint 2: Sort the intervals with respect to their starting values.
/// This will allow you to merge all overlapping intervals in a single traversal
/// through the sorted intervals.
//------------------------------------------------------------------------------
void insertion_sort_intervals(vector<vector<int>>& input // [in, out]
  )
{
  if (input.empty() || input.size() == 1)
  {
    return;
  }

  for (int k {1}; k < input.size(); ++k)
  {
    for (int j {k}; j > 0; --j)
    {
      if (input[j][0] < input[j - 1][0])
      {
        std::swap(input[j], input[j - 1]);
      }
      else if (input[j][0] == input[j - 1][0] && input[j][1] < input[j - 1][1])
      {
        std::swap(input[j], input[j - 1]);
      }
      else
      {
        // As soon as we don't need to swap, the (k + 1) element is in correct
        // location.
        break;
      }
    }
  }
}

vector<int> merge_two_intervals(
  const vector<int>& interval1,
  const vector<int>& interval2)
{
  // Fully inside cases.
  if (interval1[0] <= interval2[0] && interval2[1] <= interval1[1])
  {
    return interval1;
  }
  else if (interval2[0] <= interval1[0] && interval1[1] <= interval2[1])
  {
    return interval2;
  }
  else if (interval1[1] >= interval2[0])
  {
    return vector<int>{interval1[0], interval2[1]};
  }
  else
  {
    assert (interval2[1] >= interval1[0]);

    return vector<int>{interval2[0], interval1[1]};
  }
}

vector<vector<int>> merge_overlapping_intervals(vector<vector<int>> intervals)
{
  vector<vector<int>> sorted_intervals {intervals};

  std::sort(
    sorted_intervals.begin(),
    sorted_intervals.end(),
    [](auto interval1, auto interval2) { return interval1[0] < interval2[0]; });

  vector<vector<int>*> merged_interval_ptrs;
  vector<int>* current_interval_ptr {&sorted_intervals[0]};
  merged_interval_ptrs.emplace_back(current_interval_ptr);

  for (auto& next_interval : sorted_intervals)
  {
    const int current_interval_end {current_interval_ptr->at(1)};

    // Since we had sorted the intervals already, then the only case we need to
    // check for overlapping intervals is this one.
    if (current_interval_end >= next_interval[0])
    {
      current_interval_ptr->at(1) = std::max(
        next_interval[1],
        current_interval_end);
    }
    else
    {
      // We have a completed, merged interval.
      current_interval_ptr = &next_interval;
      merged_interval_ptrs.emplace_back(current_interval_ptr);
    }
  }

  vector<vector<int>> result {};
  for (auto& interval_ptr : merged_interval_ptrs)
  {
    result.emplace_back(*interval_ptr);
  }

  return result;
}

/* WRONG - did not think to sort beforehand.
  vector<vector<int>> result {};

  for (auto& interval : intervals)
  {
    if (result.empty())
    {
      result.emplace_back(interval);
    }
    else
    {
      vector<vector<int>> to_merge {};
      vector<vector<int>> not_to_merge {};

      for (auto& existing_interval : result)
      {
        if (is_not_overlapping(existing_interval, interval))
        {
          not_to_merge.emplace_back(existing_interval);
        }
        else
        {
          to_merge.emplace_back(existing_interval);
        }
      }

      if (to_merge.empty())
      {
        result.emplace_back(interval);
      }
      else
      {
        vector<vector<int>> new_result {};
        vector<int> merging_interval {interval};
        for (auto& interval_to_merge : to_merge)
        {
          merging_interval = merge_two_intervals(
            merging_interval,
            interval_to_merge);
        }

        new_result.emplace_back(merging_interval);

        for (auto& separate_interval : not_to_merge)
        {
          new_result.emplace_back(separate_interval);
        }

        result = new_result;
      }
    }
  }

  return result;
*/

} // namespace MergeOverlappingIntervals

} // namespace ExpertIo
} // namespace Algorithms
