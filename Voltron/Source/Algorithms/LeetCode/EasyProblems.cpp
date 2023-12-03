#include "EasyProblems.h"

#include <vector>

namespace Algorithms
{
namespace LeetCode
{

int GetMaximumInGeneratedArray::get_maximum_generated(int n)
{
  if (n == 0 || n == 1)
  {
    return n;
  }

  // Use -1 as a value to show that there wasn't a value before.
  std::vector<int> values (n + 1, -1);

  values[0] = 0;
  values[1] = 1;
  int maximum {1};

  // O(N) time complexity.
  for (int i {2}; i < n + 1; ++i)
  {
    if (values[i] == -1)
    {
      // i is even,
      if (i % 2 == 0)
      {
        values[i] = values[i / 2];
      }
      // i is odd
      else
      {
        values[i] = values[i / 2] + values[i / 2 + 1];
      }
    }

    if (values[i] > maximum)
    {
      maximum = values[i];
    }
  }

  return maximum;
}

} // namespace LeetCode
} // namespace Algorithms
