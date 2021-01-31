//------------------------------------------------------------------------------
/// \file InsertionSort.h
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating Insertion Sort.
///-----------------------------------------------------------------------------
#ifndef ALGORITHMS_SORTING_INSERTION_SORT_H
#define ALGORITHMS_SORTING_INSERTION_SORT_H

#include <algorithm> // std::swap
#include <vector>

namespace Algorithms
{
namespace Sorting
{

namespace InsertionSort
{

//------------------------------------------------------------------------------
/// \details Compare this against pseudo-code given in pp. 18, 2.1 Insertion
/// sort, Cormen, Leiserson, Rivest, and Stein (2009).
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
/// \ref 8.02 Insertion sort, U. of Waterloo, Harder (2011), ECE 250
/// Algorithms and Data Structures.
/// \details Runtime analysis:
/// In worst case, inner loop will run k times, so worst case is
///
/// N(N-1) / 2 = O(N^2).
///
/// Investigate early termination. If we pass the insertion sort a sorted list,
/// then inner condition array[j-1] > array[j] will never be true: runtime of
/// inner loop will be O(1) and runtime of insertion sort will be O(N).
///
/// Note that array[j-1] > array[j] iff (a_{j-1}, a_j) forms an inversion. The
/// swap corrects the inversion. Thus, swap will only be performed for however
/// many inversions occur.
/// If d = number of inversions, runtime is O(n+d).
//------------------------------------------------------------------------------

template <typename T>
void insertion_sort(T* const array, const int n)
{
  // body of outer for-loop will run n - 1 times with k = 1... n -1.
  for (int k {1}; k < n; ++k)
  {
    // Start from the "back" of a sorted k - 1 sized list.
    for (int j {k}; j > 0; --j)
    {
      // Body of inner loop contains if statement, all components of which run
      // in O(1) time. Because there's a break statement, use O instead of
      // Theta for the upper bound; loop may terminate early.

      if (array[j - 1] > array[j])
      {
        std::swap(array[j - 1], array[j]);
      }
      else
      {
        // As soon as we don't need to swap, the (k + 1)st is in the correct
        // location.
        break;
      }
    }
  }
}

//------------------------------------------------------------------------------
/// \ref 8.02 Insertion sort, U. of Waterloo, Harder (2011), ECE 250
/// Algorithms and Data Structures. 8.2.4 Optimizations.
/// \brief Optimized version of insertion sort, without swaps.
/// \details Instead of swapping, requiring 3 assignments, we could just assign
/// next object we're inserting into list to tmp and we only place it into the
/// array when we find its position.
//------------------------------------------------------------------------------

template <typename T>
void insertion(T* const array, int const n)
{
  for (int k {1}; k < n; ++k)
  {
    T temp {array[k]};

    for (int j {k}; j > 0; --j)
    {
      if (array[j - 1] > temp)
      {
        // "shift" or "move" the value at j - 1 into the position j.
        array[j] = array[j - 1];
      }
      else
      {
        array[j] = temp;
        break;
      }
    }

    if (array[0] > temp)
    {
      array[0] = temp; // only executed if temp < array[0]
    }
  }
}

namespace Details
{

// cf. https://www.geeksforgeeks.org/binary-insertion-sort/
// A binary search based function to find the position where item should be
// inserted in a[low ... high]
int recursive_binary_search(int a[], int item, int low, int high)
{
  if (high <= low)
  {
    return (item > a[low]) ? (low + 1) : low;
  }

  int mid = (low + high) / 2;

  if (item == a[mid])
  {
    return mid + 1;
  }

  if (item > a[mid])
  {
    return recursive_binary_search(a, item, mid + 1, high);
  }
  
  return recursive_binary_search(a, item, low, mid - 1);
}

// cf. https://www.geeksforgeeks.org/binary-insertion-sort/
// Part of Geeks for Geeks, Binary Insertion Sort

template <typename ContainerT, typename T, typename SizeT>
SizeT iterative_binary_search(
  ContainerT& a,
  const T target,
  const SizeT low,
  const SizeT high)
{
  SizeT l {low};
  SizeT h {high};
  // Assume 0 <= low and 0 <= high.
  // Consider subarray of a to include elements indexed from low to high - high,
  // but excluding a[high] element.
  // if l == h, subarray of a to consider is of size 0.

  // Includes cases of subarray of size 1 (sorted) and empty subarray.
  //if (h - 1 <= l)
  //{
  //  return (target > a[l]) ? (l + 1) : l;
  //}

  while (h - 1 > l)
  {
    // Calculate midpoint.
    // Since h - 1 > l or h - l > 1, then l <= m with m == l if  h - l < 2.
    // Thus, contradiction, l < m.
    const SizeT m {l + (h - l) / 2};

    // Since target value needs to be inserted with its duplicate, stop here.
    if (target == a[m])
    {
      // Return the new position for target to be to the "right" of the
      // duplicate value.
      return m + 1;
    }

    if (target > a[m])
    {
      l = m + 1;
    }
    // target < a[m]
    else
    {
      h = m;
    }
  }
  // Expect that h - 1 == l or h - l == 1

  return (target > a[l]) ? (l + 1) : l;
}

} // namespace Details

// cf. https://www.geeksforgeeks.org/binary-insertion-sort/
// Part of Geeks for Geeks, Binary Insertion Sort

// Time complexity is N log N for worst case; use Stirling's approximation on 
// log(1 * 2 * ... * N) = log(N!) \approx N log N
template <typename ContainerT>
void binary_insertion_sort(ContainerT& a)
{
  const std::size_t N {a.size()};
  // If k = N = 0 or 1, then a is already sorted.
  // If k = 1, k - 1 = 0. Consider k = 1, ... N - 1.
  for (std::size_t k {1}; k < N; ++k)
  {
    // Assume that 0, 1, ... k - 1 elements sorted already. Consider kth
    // element.

    // if (a[k] >= a[k - 1]), then don't do anything as it's sorted and
    // increment k. Otherwise,

    if (a[k] < a[k-1])
    {
      const auto temp = a[k];   

      std::size_t target_position {k};

      // Do a binary search for where to put it.
      // Consider the subarray that contains elements including element of
      // index l, but excluding element of index h. Thus, consider elements
      // a_l, a_{l + 1}, ... a_{k - 1}.
      //
      // Require that 0 <= l < h < N.
      // If l == h, then contradiction, since the subarray includes index l but
      // excludes index h.
      //
      // Time complexity - log(k)
      /*std::size_t l {0};
      std::size_t h {k};

      while (l < h)
      {
        // Calculate midpoint.
        // l <= m where l == m iff (h - l) < 2.
        // m <= h where m == h if h = l
        std::size_t m {l + (h - l) / 2};

        if (a[m] == temp)
        {
          target_position = m;
          break;
        }

        if (a[m] > temp)
        {
          h = m;
        }
        // if a[m] < temp
        else
        {
          l = m;
        }
      } // end of while (l < h) loop
      */

      target_position =
        Details::iterative_binary_search(a, temp, std::size_t{0}, k);

      // target_position would've only changed from original value k if it was
      // set to a value of m, where m == h if h = l. So we won't lose anything
      // in supposing we can do the following.
      //if (target_position == k)
      //{
      //  target_position = l;
      //}

      for (std::size_t index {k}; index > target_position; --index)
      {
        // "Move" values to the "right".
        a[index] = a[index - 1];
      }

      a[target_position] = temp;
    }
  } // end of for (std::size_t k {1}; k < N; ++k) loop.
}

} // namespace InsertionSort

} // namespace Sorting
} // namespace Algorithms

#endif // ALGORITHMS_SORTING_INSERTION_SORT_H