#include "Level1.h"

#include <algorithm>
#include <cstddef>
#include <map>
#include <utility> // std::swap
#include <vector>

#include <iostream>

using std::map;
using std::size_t;
using std::vector;

namespace Algorithms
{
namespace ExpertIo
{

vector<int> two_number_sum_brute(vector<int> array, int target_sum)
{
  vector<int> solution;

  // O(N) complexity.
  for (size_t i {0}; i < array.size(); ++i)
  {
    // Worse case, O(N-1) ~ O(N) complexity
    for (size_t j {i + 1}; j < array.size(); ++j)
    {
      if (array[i] + array[j] == target_sum)
      {
        solution.emplace_back(array[i]);
        solution.emplace_back(array[j]);
      }
    }
  }

  return solution;
}

vector<int> two_number_sum_with_map(vector<int> array, int target_sum)
{
  vector<int> solution;

  map<int, int> result_number;


  // O(N) time complexity. O(N) space complexity for result_number.
  for (const auto& x : array)
  {
    const int y {target_sum - x};

    result_number.insert({y, x});
  }

  // O(N) time complexity.
  for (const auto& y : array)
  {
    // Given assumption of unique numbers.
    if (result_number.contains(y) &&
      result_number.at(y) != y)
    {
      solution.emplace_back(y);
      solution.emplace_back(result_number.at(y));

      break;
    }
  }

  return solution;
}

bool is_valid_subsequence(vector<int> array, vector<int> sequence)
{
  size_t array_ptr {0};
  size_t sequence_ptr {0};

  const size_t N_arr {array.size()};
  const size_t M_seq {sequence.size()};

  // Assume N_arr >= M_seq.

  // Time complexity, O(N)
  while (array_ptr < N_arr)
  {
    // Found a match in order.
    if (array[array_ptr] == sequence[sequence_ptr])
    {
      ++sequence_ptr;
    }

    // Success condition:
    if (sequence_ptr == M_seq)
    {
      return true;
    }

    ++array_ptr;
  }

  return false;
}


vector<int> sorted_squared_array_algorithmic(vector<int> array)
{
  // https://en.cppreference.com/w/cpp/algorithm/for_each
  // Applies the given function object f to the result of dereferencing every
  // iterator in the range [first, last).

  // O(4) space.
  int array_index {0};
  int first_nonnegative_ptr {0};
  bool has_negative {false};
  const size_t N {array.size()};

  // O(N) time.
  std::for_each(
    array.begin(),
    array.end(),
    [&has_negative, &first_nonnegative_ptr, &array_index, &N](int& x)
    {
      if (!has_negative)
      {
        if (x < 0)
        {
          has_negative = true;

          first_nonnegative_ptr = N;
        }
      }
      else
      {
        if (x >= 0)
        {
          first_nonnegative_ptr = array_index;
        }
      }

      ++array_index;
      x = x * x;
    });


  if (!has_negative)
  {
    return array;
  }
  else
  {
    vector<int> sorted_array;
    // The last of the negative values.
    array_index = first_nonnegative_ptr - 1;

    while (array_index >= 0 || first_nonnegative_ptr < N)
    {
      if (array_index >= 0 && first_nonnegative_ptr < N)
      {
        if (array.at(array_index ) > array.at(first_nonnegative_ptr))
        {
          sorted_array.emplace_back(array.at(first_nonnegative_ptr));
          ++first_nonnegative_ptr;
        }
        else
        {
          sorted_array.emplace_back(array.at(array_index));
          --array_index;
        }
      }
      else
      {
        if (array_index < 0)
        {
          sorted_array.emplace_back(array.at(first_nonnegative_ptr));
          ++first_nonnegative_ptr;
        }
        else
        {
          sorted_array.emplace_back(array.at(array_index));
          --array_index;
        }
      }
    }

    return sorted_array;
  }
}

vector<int> sorted_squared_array_with_selection_sort(vector<int> array)
{
  const size_t N {array.size()};
  for (int i {0}; i < N; ++i)
  {
    array[i] = array[i] * array[i];
  }

  // Selection sort
  auto selection_sort = [](vector<int>& arr)
  {
    const size_t N {arr.size()};

    for (int i {0}; i < N; ++i)
    {
      int min_element_index {i};

      for (int j {i + 1}; j < N; ++j)
      {
        if (arr[j] < arr[min_element_index])
        {
          min_element_index = j;
        }
      }

      std::swap(arr[i], arr[min_element_index]);
    }

    return arr;
  };

  return selection_sort(array);
}

} // namespace ExpertIo
} // namespace Algorithms
