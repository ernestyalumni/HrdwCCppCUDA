//------------------------------------------------------------------------------
/// \file QuickSort.h
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating Quick Sort.
///-----------------------------------------------------------------------------
#ifndef ALGORITHMS_SORTING_QUICK_SORT_H
#define ALGORITHMS_SORTING_QUICK_SORT_H

#include <algorithm>
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

// Conquer of Divide and Conquer
std::size_t partition_from_first(
  std::vector<int>& a,
  const std::size_t left_index,
  const std::size_t right_index);

/// \param left_index is the first index in the range
template <class TContainer>
std::size_t partition_from_last(
  TContainer& a,
  const std::size_t left_index,
  const std::size_t right_index)
{
  const auto pivot_value {a[right_index]};

  std::size_t j {right_index};

  std::vector<std::size_t> range_vector(right_index - left_index);
  std::generate(
    range_vector.begin(),
    range_vector.end(),
    [i = right_index - 1]() mutable
    {
      return i--;
    });

  std::for_each(
    range_vector.begin(),
    range_vector.end(),
    [pivot_value, &a, &j](const std::size_t i)
    {
      if (a[i] >= pivot_value)
      {
        j--;
        std::swap(a[j], a[i]);
      }
    });

  std::swap(a[j], a[right_index]);

  return j;
}

//------------------------------------------------------------------------------
/// \url https://www.geeksforgeeks.org/quick-sort/
//------------------------------------------------------------------------------

// Utility function to swap 2 elements
template <typename T>
void swap_basic(T* a, T* b)
{
  T temp = *a;
  *a = *b;
  *b = temp;
}

template <typename T>
void swap_basic(T& a, T& b)
{
  T temp = a;
  a = b;
  b = temp;
}

void quick_sort_from_first(
  std::vector<int>& a,
  const std::size_t left_index,
  const std::size_t right_index);

template <class TContainer>
void quick_sort_from_last(
  TContainer& a,
  const std::size_t left_index,
  const std::size_t right_index)
{
  if (left_index < right_index + 1)
  {
    const std::size_t new_pivot {
      partition_from_last(a, left_index, right_index)};

    quick_sort_from_last(a, left_index, new_pivot - 1);
    quick_sort_from_last(a, new_pivot + 1, right_index);
  }
}

} // namespace Details

void quick_sort_from_first(std::vector<int>& a);

template <class TContainer>
void quick_sort_from_last(TContainer& a)
{
  Details::quick_sort_from_last(a, 0, a.size() - 1);
}

template <class TContainer>
ssize_t partition_from_last_basic(
  TContainer& a,
  const ssize_t left_index,
  const ssize_t right_index)
{
  // pivot.
  const auto pivot {a[right_index]};

  // Need to start position left of left_index, but left_index can be 0.
  //ssize_t i {static_cast<ssize_t>(left_index) - 1};
  //ssize_t i {left_index - 1};
  ssize_t i {left_index};

  for (ssize_t j {left_index}; j < right_index; ++j)
  {
    // If current element is smaller than the pivot, swap with what's left.
    if (a[j] < pivot)
    {
      //++i; // increment index of smaller element.
      Details::swap_basic(a[i], a[j]);
      ++i;
    }
  }

  // Swap the pivot into position such that for all indices less than this
  // position, all those elements are less than pivot.
  //Details::swap_basic(a[i + 1], a[right_index]);
  Details::swap_basic(a[i], a[right_index]);

  //return i + 1;
  return i;
}

template <typename TContainer>
void quick_sort_basic(TContainer& a, const ssize_t l, const ssize_t r)
{
  if (l < r)
  {
    // pi is partitioning index, a[pi] is now at the right place.
    ssize_t pi {partition_from_last_basic(a, l, r)};

    // Separately sort elements before the partition about the pivot and after
    // the pivot.
    quick_sort_basic(a, l, pi - 1);
    quick_sort_basic(a, pi + 1, r);
  }
}

} // namespace QuickSort

} // namespace Sorting
} // namespace Algorithms

#endif // ALGORITHMS_SORTING_QUICK_SORT_H