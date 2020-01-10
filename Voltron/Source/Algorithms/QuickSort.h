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


} // namespace QuickSort

} // namespace Sorting
} // namespace Algorithms

#endif // ALGORITHMS_SORTING_QUICK_SORT_H