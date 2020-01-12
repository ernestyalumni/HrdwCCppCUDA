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