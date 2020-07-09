//------------------------------------------------------------------------------
/// \file OnesComplementSum.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief Utilities for ones complement sum, useful for computing the Internet
/// checksum.
/// \details RFC 1071, Braden, Borman, & Patridge
/// \ref https://tools.ietf.org/html/rfc1071
/// http://kfall.net/ucbpage/EE122/lec06/tsld022.htm
//------------------------------------------------------------------------------
#ifndef ALGORITHMS_BITS_ONES_COMPLEMENT_SUM_H
#define ALGORITHMS_BITS_ONES_COMPLEMENT_SUM_H

#include <cstdint>
#include <vector>

namespace Algorithms
{
namespace Bits
{

std::uint16_t ones_complement_sum(const std::vector<uint16_t>& x);

std::uint16_t ones_complement_binary_sum(const uint16_t x, const uint16_t y);

} // namespace Bits
} // namespace Algorithms

#endif // ALGORITHMS_BITS_ONES_COMPLEMENT_SUM_H
