#ifndef ALGORITHMS_EXPERT_IO_LEVEL_1_H
#define ALGORITHMS_EXPERT_IO_LEVEL_1_H

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


} // namespace ExpertIo
} // namespace Algorithms

#endif // ALGORITHMS_EXPERT_IO_LEVEL_1_H