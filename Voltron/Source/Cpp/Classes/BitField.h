//------------------------------------------------------------------------------
/// \ref https://en.cppreference.com/w/cpp/language/bit_field
//------------------------------------------------------------------------------

#ifndef CPP_CLASSES_BIT_FIELD_H
#define CPP_CLASSES_BIT_FIELD_H

namespace Cpp
{
namespace Classes
{

struct Char4Bits
{
  char bits : 4;
};

struct Int4Bits
{
  int bits : 4;
};

struct UnsignedChar4Bits
{
  unsigned char bits : 4;
};

struct UnsignedInt4Bits
{
  unsigned int bits : 4;
};

struct UnsignedInt24Bits
{
  unsigned int bits : 24;
};

struct UnsignedInt5Bits
{
  unsigned int bits : 5;
};

} // namespace Classes
} // namespace Cpp

#endif // CPP_CLASSES_BIT_FIELD_H