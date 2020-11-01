//------------------------------------------------------------------------------
/// \file BinarySearch.h
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating Binary Search.
/// \ref 
///-----------------------------------------------------------------------------
#ifndef ALGORITHMS_SEARCH_BINARY_SEARCH_H
#define ALGORITHMS_SEARCH_BINARY_SEARCH_H

#include <cstddef> // std::size_t
#include <optional>
#include <utility> // std::make_pair

namespace Algorithms
{
namespace Search
{

namespace Details
{

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

} // namespace Search
} // namespace Algorithms

#endif // ALGORITHMS_SEARCH_BINARY_SEARCH_H