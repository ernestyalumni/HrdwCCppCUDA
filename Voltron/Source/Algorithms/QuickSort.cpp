//------------------------------------------------------------------------------
/// \file QuickSort.cpp
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating Quick Sort.
/// \ref https://stackoverflow.com/questions/22504837/how-to-implement-quick-sort-algorithm-in-c
///-----------------------------------------------------------------------------
#include <algorithm> // std::iter_swap
#include <cstddef> // std::size_t
#include <vector>


namespace Algorithms
{
namespace Sorting
{

namespace QuickSort
{

namespace Details
{

// Conquer of divide and conquer.
/// \param end_index is the first index *NOT* in the range
// \ref https://stackoverflow.com/questions/22504837/how-to-implement-quick-sort-algorithm-in-c
std::size_t partition_from_first(
  std::vector<int>& a,
  const std::size_t left_index,
  const std::size_t end_index)
{
  // Treat the chosen pivot (index and value) as an independent entity to
  // compare against and move during the entire operation.
  const int pivot_value {a[left_index]};

  std::size_t i {left_index};

  // Scan the array using another index j such that the elements at left_index
  // through i - 1 (inclusive) are less than the pivtor, and elements at i
  // through j (inclusive) are equal to or greater than pivot.
  for (std::size_t j {left_index + 1}; j < end_index; j++)
  {
    if (a.at(j) <= pivot_value)
    {
      // Originally,
      //i++;
      ++i;
      // Swap a at i and j.
      std::swap(a[i], a[j]);
    }
  }

  // Key insight is to do the swap of the pivot *after* all the other swaps that
  // ensure that all numbers to the left of the future pivot position are
  // smaller, and the other numbers to the right are all found to be larger.
  // Swap
  std::swap(a[i], a[left_index]);

  // Return the new pivot
  return i;
}

void quick_sort_from_first(
  std::vector<int>& a,
  const std::size_t left_index,
  const std::size_t end_index)
{
  if (left_index < end_index)
  {
    const std::size_t new_pivot {
      partition_from_first(a, left_index, end_index)};

    quick_sort_from_first(a, left_index, new_pivot);
    quick_sort_from_first(a, new_pivot + 1, end_index);
  }
}

} // namespace Details

void quick_sort_from_first(std::vector<int>& a)
{
  std::size_t left_index {0};
  std::size_t end_index {a.size()};

  Details::quick_sort_from_first(a, left_index, end_index);
}


} // namespace QuickSort
} // namespace Sorting
} // namespace Algorithms
