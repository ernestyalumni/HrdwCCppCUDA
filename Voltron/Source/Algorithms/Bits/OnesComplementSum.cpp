//------------------------------------------------------------------------------
/// \file OnesComplementSum.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief Utilities for bit masks.
/// \ref https://tools.ietf.org/html/rfc1071
/// http://kfall.net/ucbpage/EE122/lec06/tsld022.htm
//------------------------------------------------------------------------------
#ifndef ALGORITHMS_BITS_ONES_COMPLEMENT_SUM_H
#define ALGORITHMS_BITS_ONES_COMPLEMENT_SUM_H

#include "Algorithms/Bits/Masks.h"

#include <cstdint>
#include <numeric>
#include <vector>

using Algorithms::Bits::clear_bit;
using std::uint16_t;

namespace Algorithms
{
namespace Bits
{

uint16_t ones_complement_binary_sum(const uint16_t x, const uint16_t y)
{
  constexpr uint16_t max_limit {0xffff};

  auto z {x + y};
  const bool is_most_significant_bit_overflow {z > max_limit};
  z += is_most_significant_bit_overflow;

  if (is_most_significant_bit_overflow)
  {
    return static_cast<uint16_t>(clear_bit(z, 16));
  }
  else
  {
    return static_cast<uint16_t>(z);
  }
}

uint16_t ones_complement_sum(const std::vector<uint16_t>& x)
{
  constexpr uint16_t max_limit {0xffff};

  // Let x, y \in uint16_t
  // Let z = x + y.
  // Assume that if x + y > max, the programming language will use a "higher
  // capacity" type, namely uint32_t. Nevertheless, it's a fact that \forall
  // x, y \in uint16_t, if there is "overflow", it will at most be 1 bit in the
  // most-significant bit position + 1.

  const uint16_t sum {
    std::accumulate(
      x.begin(),
      x.end(),
      static_cast<uint16_t>(0),
      ones_complement_binary_sum)};

  return static_cast<uint16_t>(~sum);
}

} // namespace Bits
} // namespace Algorithms

#endif // ALGORITHMS_BITS_ONES_COMPLEMENT_SUM_H
