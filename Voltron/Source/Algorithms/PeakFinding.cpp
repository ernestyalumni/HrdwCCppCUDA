//------------------------------------------------------------------------------
/// \file PeakFinding.cpp
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating peak finding.
/// \ref https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/lecture-videos/MIT6_006F11_lec01.pdf
///-----------------------------------------------------------------------------
#include "PeakFinding.h"

#include <cstddef> // std::size_t

using std::size_t;

namespace Algorithms
{
namespace PeakFinding
{
namespace OneDim
{

size_t straightforward_search(const int a[], size_t N)
{
  if (a[0] >= a[1])
  {
    return 0;
  }

  if (a[N-1] >= a[N-2])
  {
    return (N- 1);
  }

  for (size_t i {1}; i < N - 1; ++i)
  {
    if (a[i] >= a[i - 1] && a[i] >= a[i + 1])
    {
      return i;
    }
  }

  return -1;
}


} // namespace OneDim
} // namespace PeakFinding
} // namespace Algorithms