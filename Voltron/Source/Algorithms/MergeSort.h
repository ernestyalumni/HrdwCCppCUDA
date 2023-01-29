//------------------------------------------------------------------------------
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating Merge Sort.
/// \ref https://gist.github.com/kbendick/1de4f311e2a780339eb3
/// O(N log(N)) time complexity, log(N) iterations, N comparisons
///-----------------------------------------------------------------------------
#ifndef ALGORITHMS_SORTING_MERGE_SORT_H
#define ALGORITHMS_SORTING_MERGE_SORT_H

#include <algorithm>
#include <cstddef> // std::size_t
#include <vector>

namespace Algorithms
{
namespace Sorting
{

namespace Details
{

template <class TContainer>
void merge(
  TContainer& a,
  const std::size_t l_low,
  const std::size_t l_high,
  const std::size_t r_low,
  const std::size_t r_high);

// Divide part of Divide & Conquer
template <class TContainer>
void merge_sort(TContainer& a, const std::size_t low, const std::size_t high)
{
  // Base case: if there is only 1 value to sort, it is already sorted.
  if (low >= high)
  {
    return;
  }

  const std::size_t mid = (low + high) / 2; // Midpoint

  merge_sort(a, low, mid); // Sort array from indices low to mid
  merge_sort(a, mid + 1, high); // Sort array from indices mid + 1 to high
  // Merge sorted subsections of the array
  merge<TContainer>(a, low, mid, mid + 1, high); 
}

template <class TContainer>
void merge(
  TContainer& a,
  const std::size_t l_low,
  const std::size_t l_high,
  const std::size_t r_low,
  const std::size_t r_high)
{
  const std::size_t L {r_high - l_low + 1};

  std::vector<typename TContainer::value_type> temp (L);

  std::size_t l {l_low};
  std::size_t r {r_low};

  for (int i {0}; i < L; ++i)
  {
    // Already done with left subarray; keep adding rest of right subarray
    if (l > l_high)
    {
      temp[i] = a[r++];
    }
    // Already done with right subarray; keep adding rest of left subarray
    else if (r > r_high)
    {
      temp[i] = a[l++];
    }
    else if (a[l] <= a[r])
    {
      temp[i] = a[l++];
    }
    else
    {
      temp[i] = a[r++];
    }
  }

  // Completed filling up temp, which is now sorted.
  /*
  for (int i {0}; i < L; ++i)
  {
    a[l_low + i] = temp[i];
  }
  */
  
  std::vector<std::size_t> range_vector(L);
  std::iota(range_vector.begin(), range_vector.end(), 0);

  std::for_each(
    range_vector.begin(),
    range_vector.end(),
    [&a, &temp, l_low](const std::size_t i)
    {
      a[l_low + i] = temp[i];
    });
}

} // namespace Details

template <class TContainer>
void merge_sort(TContainer& a)
{
  Details::merge_sort(a, 0, a.size() - 1);
}

template <class TContainer, class T>
void merge(
  TContainer& a,
  std::vector<T>& temporary,
  const std::size_t low,
  const std::size_t high);

// https://leetcode.com/problems/sort-an-array/discuss/329672/merge-sort
// cf. https://www.google.com/search?client=firefox-b-1-d&q=merge+sort+hackerrank#kpvalbx=_gy6eX9eJCPTC0PEP5IefgA024
// low = 0,
// high = N - 1
template <class TContainer, class T>
void merge_sort(
  TContainer& a,
  std::vector<T>& temporary,
  const std::size_t low,
  const std::size_t high)
{
  // Already sorted - base case.
  if (low >= high)
  {
    return;
  }

  // If (high - low) even, odd number of elements, mid is exact middle.
  // If (high - low) odd, even number of elements, mid is farthest right element
  // in "left" half.
  // Subtract to avoid overflow.
  const std::size_t mid {low + (high - low) / 2};

  merge_sort<TContainer, T>(a, temporary, low, mid);
  merge_sort<TContainer, T>(a, temporary, mid + 1, high);

  merge<TContainer, T>(a, temporary, low, high);
}

// https://leetcode.com/problems/sort-an-array/discuss/329672/merge-sort
template <class TContainer, class T>
void merge(
  TContainer& a,
  std::vector<T>& temporary,
  const std::size_t low,
  const std::size_t high)
{
  // End of the "left" halve
  std::size_t l_end {low + (high - low) / 2};
  // Start of the "right" halve
  std::size_t r_start {l_end + 1};  

  std::size_t l {low};
  std::size_t r {r_start};
  // Need an index into the temporary array.
  std::size_t temp_index {0};

  while (l <= l_end && r <= high)
  {
    // Copy over the smaller element.
    if (a[l] < a[r])
    {
      temporary[temp_index] = a[l];
      ++l;
    }
    else
    {
      temporary[temp_index] = a[r];
      ++r;
    }
    ++temp_index;
  }
  // Copy over remaining elements.
  if (l <= l_end)
  {
    for (std::size_t i {l}; i < (l_end + 1); ++i)
    {
      temporary[temp_index] = a[i];
      ++temp_index;
    }
  }
  else if (r <= high)
  {
    for (std::size_t i {r}; i < (high + 1); ++i)
    {
      temporary[temp_index] = a[i];
      ++temp_index;
    }
  }

  // Now copy sorted buffer into original array.
  for (std::size_t i {low}; i <= high; ++i)
  {
    a[i] = temporary[i-low];
  }
}

template <class TContainer, typename T>
void merge_sort_with_temp(
  TContainer& a,
  const std::size_t low,
  const std::size_t high)
{
  std::vector<T> temporary (sizeof(a) / sizeof(a[0]));

  merge_sort<TContainer, T>(a, temporary, low, high);
}

} // namespace Sorting
} // namespace Algorithms

#endif // ALGORITHMS_SORTING_MERGE_SORT_H