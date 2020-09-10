//------------------------------------------------------------------------------
// \file PrintBytes.cpp
///
/// \ref Sec. 2.1 Information Storage pp. 45, Fig. 2.4, Bryant, O'Hallaron
/// (2015)
//------------------------------------------------------------------------------
#include <cstdio>

namespace Algorithms
{
namespace Bits
{

namespace CVersion
{

typedef unsigned char* byte_pointer;

void show_bytes(byte_pointer start, size_t len)
{
  int i;
  for (i = 0; i < len; i++)
  {
    printf(" %.2x", start[i]);
  }
  printf("\n");
}

void show_int(int x)
{
  show_bytes((byte_pointer) &x, sizeof(int));
}

void show_float(float x)
{
  show_bytes((byte_pointer) &x, sizeof(float));
}

void show_pointer(void *x)
{
  show_bytes((byte_pointer) &x, sizeof(void *));
}

} // namespace CVersion

} // namespace Bits
} // namespace Algorithms