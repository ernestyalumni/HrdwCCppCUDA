#include "Level1.h"

#include <cstddef>
#include <map>
#include <vector>

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


} // namespace ExpertIo
} // namespace Algorithms
