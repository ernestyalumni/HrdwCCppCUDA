#include "Level1.h"

#include <cstddef>
#include <vector>

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


} // namespace ExpertIo
} // namespace Algorithms
