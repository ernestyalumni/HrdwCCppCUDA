#ifndef ALGORITHMS_SORTING_SELECTION_SORT_H
#define ALGORITHMS_SORTING_SELECTION_SORT_H

#include <cstddef>
#include <utility> // std::swap

namespace Algorithms
{
namespace Sorting
{

class SelectionSort
{
  public:

    SelectionSort() = default;

    //--------------------------------------------------------------------------
    /// \brief Sort a into increasing order.
    //--------------------------------------------------------------------------
    template <typename T>
    static void selection_sort(T& a)
    {
      const std::size_t N {a.size()};

      for (std::size_t i {0}; i < N; ++i)
      {
        // Exchange a[i] with smallest entry in a[i+1...N).

        // Index of minimal entry.
        std::size_t min_element_index {i};

        // \sum_{i=0}^{N} (N - i) = N(N - 1) / 2 -> O(N^2)

        for (std::size_t j {i + 1}; j < N; ++j)
        {
          if (a[j] < a[min_element_index])
          {
            min_element_index = j;
          }
        }

        std::swap(a[i], a[min_element_index]);
      }

    }

};

} // namespace Sorting
} // namespace Algorithms

#endif // ALGORITHMS_SORTING_SELECTION_SORT_H