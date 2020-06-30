//------------------------------------------------------------------------------
/// \file Masks.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief Utilities for bit masks.
/// \ref https://www.youtube.com/watch?v=NLKQEOgBAnw
/// Algorithms: Bit Manipulation. Gayle Laakmann McDowell.
//------------------------------------------------------------------------------
#ifndef ALGORITHMS_BITS_MASKS_H
#define ALGORITHMS_BITS_MASKS_H

#include <type_traits>

namespace Algorithms
{
namespace Bits
{

//------------------------------------------------------------------------------
/// \details (x & (1 << c)) != 0
//------------------------------------------------------------------------------

template <typename T>
bool is_bit_set_high(const T x, const std::size_t position)
{
  return (x & (1 << position)) != 0;
}

template <typename T>
T set_bit_high(const T x, const std::size_t position)
{
  return (x | (1 << position));
}

// cf. https://youtu.be/NLKQEOgBAnw?t=509
// Algorithms: Bit Manipulation. HackerRank.
// & (And) it with a mask with 1 but that spot (position).
template <typename T>
T clear_bit(const T x, const std::size_t position)
{
  // ~ inverts
  return (x & ~(1 << position));
}


} // namespace Bits
} // namespace Algorithms

#endif // ALGORITHMS_BITS_SHIFT_H
