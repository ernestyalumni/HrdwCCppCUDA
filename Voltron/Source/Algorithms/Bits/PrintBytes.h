//------------------------------------------------------------------------------
// \file PrintBytes.h
///
/// \ref Sec. 2.1 Information Storage pp. 45, Fig. 2.4, Bryant, O'Hallaron
/// (2015)
//------------------------------------------------------------------------------
#ifndef ALGORITHMS_BITS_SHIFT_H
#define ALGORITHMS_BITS_SHIFT_H

#include <cstdio>

namespace Algorithms
{
namespace Bits
{

namespace CVersion
{

typedef unsigned char* byte_pointer;

void show_bytes(byte_pointer start, size_t len);

void show_int(int x);

void show_float(float x);

void show_pointer(void *x);

} // namespace CVersion

} // namespace Bits
} // namespace Algorithms

#endif // ALGORITHMS_BITS_SHIFT_H
