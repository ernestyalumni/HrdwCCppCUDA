#ifndef ALGORITHMS_EXPERT_IO_LEVEL_1_H
#define ALGORITHMS_EXPERT_IO_LEVEL_1_H

#include <string>
#include <vector>

namespace Algorithms
{
namespace ExpertIo
{

std::vector<int> two_number_sum_brute(std::vector<int> array, int target_sum);

//------------------------------------------------------------------------------
/// \details Two Number Sum, Hint 3, Try storing every number in a hash table,
/// solving the equation mentioned in Hint #2 for every number, and checking if
/// the Y that you find is stored in the hash table. What are the time and space
/// implications of this approach?
//------------------------------------------------------------------------------
std::vector<int> two_number_sum_with_map(
  std::vector<int> array,
  int target_sum);

//------------------------------------------------------------------------------
/// \brief Validate Subsequence
///
/// O(N) time complexity, iterate through each element of the array (of size N).
/// 33.34 mins., stopwatch, 20210610
//------------------------------------------------------------------------------
bool is_valid_subsequence(std::vector<int> array, std::vector<int> sequence);

std::vector<int> sorted_squared_array_algorithmic(std::vector<int> array);

std::vector<int> sorted_squared_array_with_selection_sort(
  std::vector<int> array);


//------------------------------------------------------------------------------
/// \details Key observation:
/// We know that the array is already sorted. It's a strong hint that problem
/// can be solved in LINEAR TIME.
/// Also, observe that the maximum squared value from a positive number must be
/// on the "right" of the array, and max. squared value from a negative number
/// must be on the left, due to given a sorted array.
//------------------------------------------------------------------------------
std::vector<int> sorted_squared_array_two_indices(std::vector<int> array);

//------------------------------------------------------------------------------
/// \name Tournament Winner
/// \details For N teams, N choose 2 (\binom{N}{2}) for total number of games.
//------------------------------------------------------------------------------

std::string tournament_winner(
  std::vector<std::vector<std::string>> competitions,
  std::vector<int> results);

namespace EasyNonConstructibleChange
{

//------------------------------------------------------------------------------
/// \name Non-Constructible Change
//------------------------------------------------------------------------------
int non_constructible_change_sort(std::vector<int> coins);

/*
int min_unavailable_change(
  int& min_change,
  int& smallest_value,
  std::vector<int>& coins_left);
*/

} // namespace EasyNonConstructibleChange

} // namespace ExpertIo
} // namespace Algorithms

#endif // ALGORITHMS_EXPERT_IO_LEVEL_1_H