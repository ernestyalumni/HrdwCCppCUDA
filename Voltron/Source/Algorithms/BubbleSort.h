//------------------------------------------------------------------------------
/// \file BubbleSort.h
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating Bubble Sort.
///-----------------------------------------------------------------------------
#ifndef ALGORITHMS_SORTING_BUBBLE_SORT_H
#define ALGORITHMS_SORTING_BUBBLE_SORT_H

#include <algorithm> // std::for_each
#include <array>
#include <cassert>
#include <cstddef>
#include <numeric> // std::iota
#include <vector>

namespace Algorithms
{
namespace Sorting
{

namespace Details
{

template <typename T, std::size_t N>
void single_swap(std::array<T, N>& a, std::size_t l, std::size_t r)
{
  assert(a.size() > l && a.size() > r);

  if (l == r)
  {
    return;
  }

  const T temp {a.at(l)};

  a[l] = a[r];
  a[r] = temp;
}

template <template <class ...> class DefinedContainer, class T>
void single_swap(DefinedContainer<T>& a, std::size_t l, std::size_t r)
{
  assert(a.size() > l && a.size() > r);

  if (l == r)
  {
    return;
  }

  const T temp {a.at(l)};

  a[l] = a[r];
  a[r] = temp; 
}

//template <template <class ...> class DefinedContainer, class T, std::size_t TN>
template <class TContainer>
bool naive_single_pass(TContainer& a)
{
  bool is_swapped {false};

  const std::size_t N {a.size()};

  std::vector<std::size_t> range_vector(N - 1);
  // https://en.cppreference.com/w/cpp/algorithm/iota
  // template <class ForwardIt, class T>
  // void iota(ForwardIt first, ForwardIt last, T value);
  // Fills the range [first, last) with sequentially increasing values, starting
  // with value and repetitively evaluating ++value.
  std::iota(range_vector.begin(), range_vector.end(), 1);

  auto comparison_swap = [&a, &is_swapped](const std::size_t i)
  {
    if (a[i - 1] > a[i])
    {
      // Swap them and remember something changed
      single_swap(a, i - 1, i);
      is_swapped = true;
    }
  };

  std::for_each(range_vector.begin(), range_vector.end(), comparison_swap);

  return is_swapped;
}

template <class TContainer>
bool single_pass(TContainer& a, const std::size_t j)
{
  bool is_swapped {false};

  const std::size_t N {a.size()};

  std::vector<std::size_t> range_vector(N - j);

  std::iota(range_vector.begin(), range_vector.end(), 1);

  auto comparison_swap = [&a, &is_swapped](const std::size_t i)
  {
    if (a[i - 1] > a[i])
    {
      // Swap them and remember something changed
      single_swap(a, i - 1, i);
      is_swapped = true;
    }
  };

  std::for_each(range_vector.begin(), range_vector.end(), comparison_swap);

  return is_swapped;
}

} // namespace Details

// cf. https://en.wikipedia.org/wiki/Bubble_sort
template <class TContainer>
void naive_bubble_sort(TContainer& a)
{
  bool is_swapped {false};

  is_swapped = Details::naive_single_pass(a);

  while (is_swapped)
  {
    is_swapped = Details::naive_single_pass(a);
  }
}

// cf. https://en.wikipedia.org/wiki/Bubble_sort
// Optimized bubble sort by observing the jth pass finds the jth largest element
// and puts it into its final place. So the inner loop can avoid looking at the
// last j-1 items.
template <class TContainer>
void bubble_sort(TContainer& a)
{
  bool is_swapped {false};

  unsigned int j {1};

  is_swapped = Details::single_pass(a, j);

  j++;

  while (is_swapped)
  {
    is_swapped = Details::single_pass(a, j);
    j++;
  }
}


} // namespace Sorting
} // namespace Algorithms

#endif // ALGORITHMS_SORTING_BUBBLE_SORT_H