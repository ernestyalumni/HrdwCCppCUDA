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


} // namespace Search
} // namespace Algorithms
