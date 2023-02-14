#ifndef ALGORITHMS_SORTING_INSERTION_SORT_H
#define ALGORITHMS_SORTING_INSERTION_SORT_H

#include <algorithm> // std::swap

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
void insertion_sort_optimized(T* const array, int const n)
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

} // namespace InsertionSort

} // namespace Sorting
} // namespace Algorithms

#endif // ALGORITHMS_SORTING_INSERTION_SORT_H