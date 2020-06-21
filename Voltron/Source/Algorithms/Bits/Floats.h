//------------------------------------------------------------------------------
/// \file Floats.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief Utilities to deal with floats.
/// \ref
//------------------------------------------------------------------------------
#ifndef ALGORITHMS_BITS_FLOATS_H
#define ALGORITHMS_BITS_FLOATS_H

#include <type_traits> // std::enable_if_t

namespace Algorithms
{
namespace Bits
{

template <
  typename FP,
  std::enable_if_t<std::is_floating_point<FP>::value, int> = 0
  >
auto floating_point_to_bytes(FP value)
{
  
}

} // namespace Bits
} // namespace Algorithms

#endif // ALGORITHMS_BITS_FLOATS_H
