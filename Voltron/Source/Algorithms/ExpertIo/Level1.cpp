#include "Level1.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <map>
#include <optional>
#include <unordered_map>
#include <utility> // std::swap
#include <vector>

using std::abs;
using std::map;
using std::nullopt;
using std::optional;
using std::size_t;
using std::string;
using std::unordered_map;
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

//------------------------------------------------------------------------------
/// \details Two Number Sum Hint 2 Realize that for every number X in the input
/// array, you are essentially trying to find a corresponding number Y such that
/// X + Y = targetSum. With two variables in this equation known to you, it
/// shouldn't be hard to solve for Y.
//------------------------------------------------------------------------------

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

vector<int> two_number_sum_with_sorting(vector<int> array, int target_sum)
{
  // Doing this now allows us to use the condition that for an array
  // a_0, a_1, ... a_{N-1}, then
  // a_0 <= a_1 <= ... <= a_{N-1}
  std::sort(array.begin(), array.end());

  int left_index {0};
  int right_index {static_cast<int>(array.size()) - 1};

  while (left_index < right_index)
  {
    // If a_l + a_r = T, then we're done.
    if (array[left_index] + array[right_index] == target_sum)
    {
      return vector<int>{array[left_index], array[right_index]};
    }
    // if a_l + a_r > T, set r = r - 1 (moving to the right index by 1 will only
    // make the possible sum decrease monotonically).
    else if (array[left_index] + array[right_index] > target_sum)
    {
      right_index -= 1;
    }
    else
    {
      assert(array[left_index] + array[right_index] < target_sum);
      left_index += 1;
    }
  }

  return vector<int>{};
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

vector<int> sorted_squared_array_two_indices(vector<int> array)
{
  vector<int> results (array.size(), 0);

  // Index "pointer" along results.
  //int right_to_left_index {static_cast<int>(array.size() - 1)};

  // Index "pointer" for possibly positive numbers. We know that in a sorted
  // array, the positive numbers, if any, must start from the "right"
  int right_index {static_cast<int>(array.size() - 1)};

  // Index "pointer" for possibly negative numbers. Key observation was that we
  // know in a sorted array, negative numbers, if any, must start from the
  // "left."
  int left_index {0};

  // Don't need any of this.
  /*
  if (array[right_index] < 0)
  {
    right_index = -1;
  }

  if (array[left_index] >= 0)
  {
    left_index = array.size();
  }

  while (right_to_left_index >= 0)
  {
    optional<int> right_value;

    if (right_index >= 0 && array[right_index] >= 0)
    {
      right_value = array[right_index];
    }
    else
    {
      right_value = nullopt;
    }

    optional<int> left_value;

    if (left_index < array.size() && array[left_index] < 0)
    {
      left_value = abs(array[left_index]);
    }
    else
    {
      left_value = nullopt;
    }

    //left_value =
      //left_index < array.size() && array[left_index] < 0 ?
        //abs(array[left_index]) : nullopt;

    if (right_value.has_value() && left_value.has_value())
    {
      if (right_value.value() >= left_value.value())
      {
        results[right_to_left_index] =
          right_value.value() * right_value.value();
        --right_index;
      }
      else
      {
        results[right_to_left_index] =
          left_value.value() * left_value.value();
        --left_index;        
      }
    }
    else if (right_value.has_value())
    {
      results[right_to_left_index] =
        right_value.value() * right_value.value();
      --right_index;
    }
    else if (left_value.has_value())
    {
      results[right_to_left_index] =
        left_value.value() * left_value.value();
      --left_index;        
    }


    --right_to_left_index;
  }
  */

  for (int index {static_cast<int>(array.size() - 1)}; index >= 0; --index)
  {
    const int left_value {array[left_index]};
    const int right_value {array[right_index]};

    // Don't forget to put absolute value sign; otherwise, first, it would
    // defeat purpose of comparison (negative values always less than positive)
    // values, and second, remember we want the squared values to be sorted.
    if (abs(left_value) > abs(right_value))
    {
      results[index] = left_value * left_value;
      ++left_index;
    }
    else
    {
      results[index] = right_value * right_value;
      --right_index;
    }
  }

  return results;
}

string tournament_winner(
  vector<vector<string>> competitions,
  vector<int> results)
{
  // Implemented as hash map. Has constant time access.
  unordered_map<string, int> scores;

  assert(results.size() == competitions.size());

  const int M {static_cast<int>(competitions.size())};

  for (int i {0}; i < M; ++i)
  {
    const string winner {
      results.at(i) == 1 ? competitions.at(i).at(0) : competitions.at(i).at(1)};

    // Constant time in search for the key, in this case, which is winner.
    if (scores.count(winner) == 0)
    {
      // Team not yet accounted for case.
      scores[winner] = 3;
    }
    else
    {
      // Team already has scores. Add to it.
      scores[winner] += 3;
    }
  }

  string winning_team;
  int max_score {0};


  // O(N) time. N number of teams.
  for (auto iter {scores.begin()}; iter != scores.end(); ++iter)
  {
    if (iter->second >= max_score)
    {
      winning_team = iter->first;
      max_score = iter->second;
    }
  }

  return winning_team;
}

namespace EasyNonConstructibleChange
{

int non_constructible_change_sort(vector<int> coins)
{
  if (coins.size() == 0)
  {
    return 1;
  }

  // O(N log N) comparisons.
  std::sort(coins.begin(), coins.end());    

  int sum_counter {0};
  int coin_type_counter {1};

  for (auto& coin : coins)
  {
    assert(coin >= coin_type_counter);    

    if (coin > coin_type_counter)
    {
      if (coin == coin_type_counter + 1)
      {
        coin_type_counter = coin;
        sum_counter += coin;
      }
      else
      {
        return sum_counter + 1;
      }
    }

    if (coin == coin_type_counter)
    {
      sum_counter += coin;
    }
  }

  return sum_counter + 1;
}

/*
int min_unavailable_change(
  int& min_unavailable_change,
  int& smallest_value,
  vector<int>& coins_left)
{
  if (coins_left.size() == 0)
  {
    return 1;
  }

  if (coins_left.size() == 1)
  {
    const int last_coin {coins_left.back()};

    if (last_coin == 1)
    {
      return min_unavailable_change + last_coin;
    }

    return min_unavailable_change;
  }

  const int next_coin {coins_left.back()};
  coins_left.pop();

  if (next_coin == 1)
  {
    min_unavailable_change += next_coin;
  }

  if (next_coin <= min_unavailable_change)
  {

  }
}
*/

} // namespace EasyNonConstructibleChange

vector<vector<int>> transpose_matrix(vector<vector<int>> matrix)
{
  const size_t M {matrix.size()};
  const size_t N {matrix[0].size()};

  // O(M*N) space complexity.
  vector<vector<int>> transposed_matrix {};

  // O(M*N) time complexity.
  for (size_t i {0}; i < N; ++i)
  {
    vector<int> row {};
    for (size_t j {0}; j < M; ++j)
    {
      row.push_back(matrix[j][i]);
    }

    transposed_matrix.push_back(row);
  }

  return transposed_matrix;
}

namespace NthFibonacci
{

int get_nth_fibonacci_recursive(const int n)
{
  if (n == 0 || n == 1)
  {
    return 0;
  }

  if (n == 2)
  {
    return 1;
  }

  return get_nth_fibonacci_recursive(n - 1) +
    get_nth_fibonacci_recursive(n - 2);
}

} // namespace NthFibonacci

} // namespace ExpertIo
} // namespace Algorithms
