//------------------------------------------------------------------------------
/// \file BinarySearch.h
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating Binary Search.
//------------------------------------------------------------------------------
#ifndef ALGORITHMS_SEARCH_BINARY_SEARCH_H
#define ALGORITHMS_SEARCH_BINARY_SEARCH_H

#include <cstddef> // std::size_t
#include <optional>
#include <utility> // std::make_pair

#include <iostream>

namespace Algorithms
{
namespace Search
{

//------------------------------------------------------------------------------
/// \ref Exercise 2.3-5, Cormen, Leiserson, Rivest, and Stein (2009)
//------------------------------------------------------------------------------
template <typename T, typename ContainerT>
std::optional<std::size_t> binary_search_iterative(
  const ContainerT& a,
  const T target,
  //----------------------------------------------------------------------------
  /// \details Be careful of a huge error case that gets caught by a unit test
  /// testing a value smaller than the left bound. The exit condition requires
  /// that l > r. But what happens if l = 0 and the the type is unsigned
  /// std::size_t? Underflow!
  //----------------------------------------------------------------------------
  const int high,
  const int low = 0)
{
  int r {high};
  int l {low};

  while (l <= r)
  {
    // Gets highest value in left half; lies on "left side" of middle pt. if
    // number of elements is even.
    const int mid {(l + r) / 2};

    if (a[mid] == target)
    {
      return mid;
    }

    if (a[mid] < target)
    {
      l = mid + 1;
    }
    else
    {
      r = mid - 1;
    }
  }

  return std::nullopt;
}

template <typename T, typename ContainerT>
std::optional<std::size_t> binary_search_recursive(
  const ContainerT& a,
  const T target,
  const int high,
  const int low = 0)
{
  if (low > high)
  {
    return std::nullopt;
  }

  const int mid {(low + high) / 2};

  if (a[mid] == target)
  {
    return mid;
  }

  if (a[mid] < target)
  {
    return binary_search_recursive(a, target, high, mid + 1);
  }
  else
  {
    return binary_search_recursive(a, target, mid - 1, low);
  }
}

// cf. https://web2.qatar.cmu.edu/~mhhammou/15122-s16/lectures/06-binsearch.pdf
// Pfenning (2016).

template <typename T, typename SizeT>
std::optional<SizeT> binary_search(T x, T* a, SizeT n)
{
  // assert 0 <= n && require n <= sizeof A / sizeof A[0]
  // assert is_sorted(A, 0, n)
  // ensure nullopt && !is_in(x, A, 0, n) || (0 <= result < n) && A[result] == x

  // Subarray to consider is indexed from low to high - 1
  // (i.e. subarray includes low and excludes high)
  
  SizeT low {0};
  SizeT high {n};

  // if low == high, then the subarray to consider is of size 0 since by
  // contradiction the subarray must include low and exclude high.
  while (low < high)
  {
    // Calculate midpoint. For (high - low) being even, this will land you to
    // the start of the "right half", "second" partition/half. 
    const SizeT m {low + (high - low) / 2};

    if (a[m] == x)
    {
      return m;
    }

    if (a[m] < x)
    {
      low = m + 1;
    }
    // a[m] > x
    else
    {
      high = m;
    }
  }

  return std::nullopt;
}

/// Runtime: 72 ms
/// Memory Usage: 27.9 MB
template <typename ContainerT, typename T>
std::optional<std::size_t> binary_search_inclusive(
  const ContainerT& a,
  const T target_value)
{
  std::size_t l {0};
  std::size_t r {a.size() - 1}; 

  // if l > r, subarray contains no elements.
  // 0 <= l <= r <= N - 1 < N 
  while (l <= r)
  {
    // Calculate midpoint. (floor of (high + low) / 2)
    // l <= m <= r
    const std::size_t m {l + (r - l + 1) / 2};

    if (a[m] == target_value)
    {
      return m;
    }

    if (a[m] > target_value)
    {
      // std::size_t type forces value to be non-negative.
      if (m > 0)
      {
        r = m - 1;
      }
      else
      {
        return std::nullopt;
      }
    }
    // a[m] < target_value
    else
    {
      l = m + 1;
    }
  }

  return std::nullopt;
}

//------------------------------------------------------------------------------
/// \url https://leetcode.com/explore/learn/card/binary-search/125/template-i/950/
//------------------------------------------------------------------------------

int square_root(int x);


//------------------------------------------------------------------------------
/// \url https://leetcode.com/discuss/interview-question/algorithms/124724/facebook-onsite-count-occurrences-of-a-number-in-a-sorted-array
/// \ref https://www.geeksforgeeks.org/count-number-of-occurrences-or-frequency-in-a-sorted-array/
//------------------------------------------------------------------------------

template <
  typename T,
  typename FindFirstT,
  typename FindLastT,
  typename ContainerT>
int find_number_of_occurrences_in_sorted_array(
  FindFirstT f1,
  FindLastT f2,
  const ContainerT& a,
  const T target,
  const int high,
  const int low = 0)
{
  const auto result1 = f1(a, target, high, low);

  if (!static_cast<bool>(result1))
  {
    return 0;
  }

  return f2(a, target, high, low).value() - result1.value() + 1;
}

namespace Details
{

//------------------------------------------------------------------------------
/// \url https://leetcode.com/discuss/interview-question/algorithms/124724/facebook-onsite-count-occurrences-of-a-number-in-a-sorted-array
//------------------------------------------------------------------------------

template <typename T, typename ContainerT>
std::optional<std::size_t> binary_search_first_occurrence_recursive(
  const ContainerT& a,
  const T target,
  const int high,
  const int low = 0)
{
  if (low > high)
  {
    return std::nullopt;
  }

  const int mid {(low + high) / 2};

  if ((mid == 0 || a[mid - 1] < target) && (a[mid] == target))
  {
    return mid;
  }
  else if (a[mid] < target)
  {
    return binary_search_first_occurrence_recursive(a, target, high, mid + 1);
  }
  else
  {
    return binary_search_first_occurrence_recursive(a, target, mid - 1, low);
  }
}

template <typename T, typename ContainerT>
std::optional<std::size_t> binary_search_first_occurrence_iterative(
  const ContainerT& a,
  const T target,
  const int high,
  const int low = 0)
{
  int r {high};
  int l {low};

  while (l <= r)
  {
    const int mid {(l + r) / 2};

    if ((mid == 0 || a[mid - 1] < target) && (a[mid] == target))
    {
      return mid;
    }
    else if (a[mid] < target)
    {
      l = mid + 1;
    }
    else
    {
      r = mid - 1;
    }
  }

  return std::nullopt;
}

template <typename T, typename ContainerT>
std::optional<std::size_t> binary_search_last_occurrence_recursive(
  const ContainerT& a,
  const T target,
  const int high,
  const int low = 0)
{
  const int N {static_cast<int>(a.size())};

  if (low > high)
  {
    return std::nullopt;
  }

  const int mid {(low + high) / 2};

  if ((mid == N - 1 || a[mid + 1] > target) && (a[mid] == target))
  {
    return mid;
  }
  else if (a[mid] > target)
  {
    return binary_search_last_occurrence_recursive(a, target, mid - 1, low);
  }
  else
  {
    return binary_search_last_occurrence_recursive(a, target, high, mid + 1);
  }
}

template <typename T, typename ContainerT>
std::optional<std::size_t> binary_search_last_occurrence_iterative(
  const ContainerT& a,
  const T target,
  const int high,
  const int low = 0)
{
  int r {high};
  int l {low};

  const int N {static_cast<int>(a.size())};

  while (l <= r)
  {
    const int mid {(l + r) / 2};

    if ((mid == N - 1 || a[mid + 1] > target) && (a[mid] == target))
    {
      return mid;
    }
    else if (a[mid] > target)
    {
      r = mid - 1;
    }
    else
    {
      l = mid + 1;
    }
  }

  return std::nullopt;
}

//------------------------------------------------------------------------------
/// Not great code.
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
/// \details If l - r odd so that total number of elements is even, midpoint is
/// the position of the "farthest right" element in the "left" half.
//------------------------------------------------------------------------------

std::optional<std::size_t> calculate_midpoint(
  const std::size_t l,
  const std::size_t r);

template <typename T>
std::optional<std::pair<bool, std::pair<std::size_t, std::size_t>>>
  compare_partition(
    const T& midpoint_value,
    const T& search_value,
    const std::size_t midpoint_index,
    const std::size_t l,
    const std::size_t r)
{
  if (search_value == midpoint_value)
  {
    return 
      std::make_pair(
        true,
        std::make_pair(midpoint_index, midpoint_index));
  }

  if (search_value < midpoint_value)
  {
    if (midpoint_index == 0)
    {
      return std::nullopt;
    }
    return std::make_pair(false, std::make_pair(l, midpoint_index - 1));
  }

  //if (search_value > midpoint_value)
  //{
  if (midpoint_index + 1 > r)
  {
    return std::nullopt;
  }
  return std::make_pair(false, std::make_pair(midpoint_index + 1, r));
  //}
}

template <class TContainer, typename T>
std::optional<std::pair<bool, std::pair<std::size_t, std::size_t>>>
  binary_search_iteration(
    const TContainer& a,
    std::size_t l,
    std::size_t r,
    const T& search_value)
{
  const auto m = calculate_midpoint(l, r);

  if (!m)
  {
    return std::nullopt;
  }

  return compare_partition(a[m.value()], search_value, m.value(), l, r);
}

} // namespace Details

template <class TContainer, typename T>
std::optional<std::size_t>
  binary_search(const TContainer& a, const T& search_value)
{
  std::size_t l {0};
  std::size_t r {a.size() - 1};

  while (true)
  {
    const auto result = Details::binary_search_iteration(a, l, r, search_value);

    if (!result)
    {
      return std::nullopt;
    }

    // Found search value!
    if (result.value().first)
    {
      return result.value().second.first;
    }

    l = result.value().second.first;
    r = result.value().second.second;
  }
}

} // namespace Search
} // namespace Algorithms

#endif // ALGORITHMS_SEARCH_BINARY_SEARCH_H