#include "PageFaults.h"

#include <algorithm>
#include <vector>

namespace QuestionsDEntrevue
{

int page_faults(const int n, const int c, int pages[])
{
  // Initialize count to 0.
  int count {0};

  // To store elemtns in memory of size c.
  std::vector<int> v;
  for (int i {0}; i < n; ++i)
  {
    // Find if element is present in memory or not.
    auto it = std::find(v.begin(), v.end(), pages[i]);

    // If element isn't present
    if (it == v.end())
    {
      // If memory is full
      if (v.size() == c)
      {
        // Remove the first element as it is least recently used.
        v.erase(v.begin());
      }

      // Add the recent element into memory.
      v.push_back(pages[i]);

      // Increment the count.
      count++;
    }
    else
    {
      // If the element is present, remove the element and add it at the end as
      // it's the most recent element.
      v.erase(it);
      v.push_back(pages[i]);
    }
  }

  return count;
}

} // namespace QuestionsDEntrevue

