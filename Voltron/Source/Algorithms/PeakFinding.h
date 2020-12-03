//------------------------------------------------------------------------------
/// \file PeakFinding.h
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating peak finding.
/// \ref https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/lecture-videos/MIT6_006F11_lec01.pdf
///-----------------------------------------------------------------------------
#ifndef ALGORITHMS_PEAK_FINDING_H
#define ALGORITHMS_PEAK_FINDING_H

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

} // namespace OneDim
} // namespace PeakFinding
} // namespace Algorithms

#endif // ALGORITHMS_PEAK_FINDING_H