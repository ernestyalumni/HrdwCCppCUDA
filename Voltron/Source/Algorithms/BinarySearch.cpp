//------------------------------------------------------------------------------
/// \file BinarySearch.cpp
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating Binary Search.
/// \ref 
///-----------------------------------------------------------------------------
#include "BinarySearch.h"

#include <cstddef> // std::size_t
#include <optional>

namespace Algorithms
{
namespace Search
{

namespace Details
{

std::optional<std::size_t> calculate_midpoint(
  const std::size_t l,
  const std::size_t r)
{
  if (r < l)
  {
    return std::nullopt;
  }

  const std::size_t L {r - l + 1};

  return (L % 2 == 1) ? (L / 2 + l) : (L / 2 - 1 + l);
}

} // namespace Details

/// Runtime: 4 ms
/// Memory Usage: 6.4 MB

int square_root(int x)
{
  if (x == 0)
  {
    return 0;
  }

  int l {1};
  int r {x};

  /*

  while (l <= r)
  {
    // If (l - r) odd, total number of elements to consider is even, and mid
    // will be "farthest right" element in "left" half.
    // will be "most left" element in "right" half.
    int mid {l + (r - l + 1) / 2};

    if (mid * mid == x)
    {
      return mid;
    }

    if (mid * mid < x)
    {
      l = mid;
      // Because we don't want to return exact match but the fact that we get a
      // greatest upper bound.
      if (r * r == x)
      {
        return r;
      }

      if (r * r > x)
      {
        --r;
      }
    }
    else
    {
      // Don't try numbers that exceed the value of x when squared.
      r = mid - 1;
    }
  }

  return l;
  */

  while (l <= r)
  {
    int mid {l + (r - l) / 2};

    if (mid > x / mid)
    {
      r = mid - 1;
    }
    else
    {
      // If mid is just at the "cusp" of being the solution,
      if (mid + 1 > x / (mid + 1))
      {
        return mid;
      }

      l = mid + 1;
    }
  }

  return l;
}

} // namespace Search
} // namespace Algorithms
