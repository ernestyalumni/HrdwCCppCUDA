//------------------------------------------------------------------------------
/// \file PeakFinding.h
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating peak finding.
/// \ref https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/lecture-videos/MIT6_006F11_lec01.pdf
///-----------------------------------------------------------------------------
#ifndef ALGORITHMS_PEAK_FINDING_H
#define ALGORITHMS_PEAK_FINDING_H

#include <cassert>
#include <cstddef> // std::size_t
#include <utility> // std::make_pair

namespace Algorithms
{
namespace PeakFinding
{
namespace OneDim
{

std::size_t straightforward_search(const int a[], std::size_t N);

template <typename ContainerT>
std::size_t straightforward_search(ContainerT a)
{
  if (a.at(0) >= a.at(1))
  {
    return 0;
  }

  if (a.at(a.size() - 1) >= a.at(a.size() - 2))
  {
    return a.size() - 1;
  }

  for (std::size_t i {1}; i < a.size() - 1; ++i)
  {
    if (a.at(i) >= a.at(i - 1) && a.at(i) >= a.at(i + 1))
    {
      return i;
    }
  }

  return -1;
}

/*
template <typename ContainerT>
std::size_t binary_search_iterative(
  ContainerT a,
  const std::size_t midpoint_index,
  const std::size_t N)
{
  if (midpoint_index == 0)
  {
    return (a[midpoint_index] >= a[midpoint_index + 1]) ?
      midpoint_index : midpoint_index + 1;
  }

  if (midpoint_index == N - 1)
  {
    return (a[midpoint_index] >= a[midpoint_index - 1]) ?
      midpoint_index : midpoint_index - 1;
  }

  if (a[midpoint_index] >= a[midpoint_index - 1]) &&
    (a[midpoint_index] >= a[midpoint_index + 1])
  {
    return midpoint_index;
  }
  else if (a[midpoint_index] < a[midpoint_index - 1])
  {
    return midpoint_index / 2;
  }
  // a[midpoint_index] < a[midpoint_index + 1]
  else
  {
    // (N - midpoint_index - 1) is the number of elements to consider
    return (N)
  }
}
*/

template <typename ContainerT>
std::size_t binary_search_recursive(
  ContainerT a,
  const std::size_t start_index,
  const std::size_t end_index,
  const std::size_t N)
{
  assert(start_index <= end_index);
  assert(start_index < N && end_index < N);

  if (start_index == 0 && end_index == 1)
  {
    return a[start_index] >= a[end_index] ? start_index :
      end_index < N ? end_index : -1;
  }

  if (start_index == N - 2 && end_index == N - 1)
  {
    return a[start_index] >= a[end_index] ? start_index : end_index;
  }

  const std::size_t midpoint_index {
    (end_index - start_index) / 2 + start_index};

  if (a[midpoint_index] >= a[midpoint_index - 1] &&
    a[midpoint_index] >= a[midpoint_index + 1])
  {
    return midpoint_index;
  }
  else if (a[midpoint_index] < a[midpoint_index - 1])
  {
    return binary_search_recursive<ContainerT>(
      a,
      start_index,
      midpoint_index - 1,
      N);
  }
  // a[midpoint_index] < a[midpoint_index + 1]
  else
  {
    return binary_search_recursive<ContainerT>(
      a,
      midpoint_index + 1,
      end_index,
      N);
  }
}



} // namespace OneDim
} // namespace PeakFinding
} // namespace Algorithms

#endif // ALGORITHMS_PEAK_FINDING_H