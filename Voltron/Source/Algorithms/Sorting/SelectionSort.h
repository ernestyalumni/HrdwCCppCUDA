#ifndef ALGORITHMS_SORTING_SELECTION_SORT_H
#define ALGORITHMS_SORTING_SELECTION_SORT_H

#include <cstddef>
#include <limits>
#include <utility> // std::swap

namespace Algorithms
{
namespace Sorting
{

namespace SelectionSort
{


class SelectionSortIterative
{
  public:

    SelectionSortIterative() = default;

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

//------------------------------------------------------------------------------
/// \brief Returns index of minimum value in array
/// array[lower_index...end_index]
//------------------------------------------------------------------------------
template <typename A, typename T>
std::size_t minimum_value_index(
  const A array,
  const std::size_t lower_index,
  const std::size_t upper_index)
{
  T min_value {std::numeric_limits<T>::max()};
  std::size_t min_index {lower_index};

  for (std::size_t i {lower_index}; i < upper_index; ++i)
  {
    if (min_value > array[i])
    {
      min_value = array[i];
      min_index = i;
    }
  }

  return min_index;
}

// cf. https://www.geeksforgeeks.org/practice-questions-for-recursion/
template <typename A, typename T>
void selection_sort_recursive(
  A& array,
  const std::size_t start_index,
  const std::size_t end_index)
{
  if (start_index >= end_index)
  {
    return;
  }

  std::size_t min_index;
  T temp_value;

  min_index = minimum_value_index<A, T>(array, start_index, end_index);

  std::swap(array[start_index], array[min_index]);

  selection_sort_recursive<A, T>(array, start_index + 1, end_index);
}


} // namespace SelectionSort

} // namespace Sorting
} // namespace Algorithms

#endif // ALGORITHMS_SORTING_SELECTION_SORT_H