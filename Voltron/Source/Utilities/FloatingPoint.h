//------------------------------------------------------------------------------
/// \file FloatingPoint.h
/// \author Ernest Yeung
/// \brief Dealing with floating point representations.
/// \ref 
///-----------------------------------------------------------------------------
#ifndef UTILITIES_FLOATING_POINT_H
#define UTILITIES_FLOATING_POINT_H

#include "Cpp/Utilities/SuperBitSet.h"

// cf. https://en.cppreference.com/w/cpp/types/climits
// Number of bits in a byte.
#include <climits> // CHAR_BIT

namespace Utilities
{

namespace Conversions
{

namespace WithUnion
{

//------------------------------------------------------------------------------
/// \class FloatingPointToBitsInUnion
/// \ref https://stackoverflow.com/questions/474007/floating-point-to-binary-valuec
//------------------------------------------------------------------------------
class FloatingPointToBitsInUnion
{
  public:

    // cf. https://en.cppreference.com/w/cpp/language/union
    // Union class type can hold only 1 of its non-static data members at a
    // time. Union is only as big as necessary to hold its largest data member.
    // Other data members are allocated in same bytes as part of that largest
    // member.
    // It's undefined behavior to read from member of union that wasn't most
    // recently written.
    union FloatUIntDirectSum
    {
       float input_;
       unsigned int output_;
    };

    union DoubleULongDirectSum
    {
       double input_;
       unsigned long output_;
    };
};

//------------------------------------------------------------------------------
/// \ref https://stackoverflow.com/questions/474007/floating-point-to-binary-valuec
//------------------------------------------------------------------------------
template <typename FP, std::size_t NumberOfBytes>
Cpp::Utilities::SuperBitSet<NumberOfBytes * CHAR_BIT> floating_point_to_bitset(
  const FP);

} // namespace WithUnion

namespace WithMemCpy
{

// TODO: Do floating-point conversion to binary by other ways. Particularly,
// using memcpy
// cf. https://stackoverflow.com/questions/48041521/showing-binary-representation-of-floating-point-types-in-c
/*

static_assert(sizeof(float) == sizeof(uint32_t));
static_assert(sizeof(double) == sizeof(uint64_t));

std::string as_binary_string( float value ) {
    std::uint32_t t;
    std::memcpy(&t, &value, sizeof(value));
    return std::bitset<sizeof(float) * 8>(t).to_string();
}

std::string as_binary_string( double value ) {
    std::uint64_t t;
    std::memcpy(&t, &value, sizeof(value));
    return std::bitset<sizeof(double) * 8>(t).to_string();
}

You may need to change the helper variable t in case the sizes for the floating point numbers are different.

You can alternatively copy them bit-by-bit. This is slower but serves for arbitrarily any type.

template <typename T>
std::string as_binary_string( T value )
{
    const std::size_t nbytes = sizeof(T), nbits = nbytes * CHAR_BIT;
    std::bitset<nbits> b;
    std::uint8_t buf[nbytes];
    std::memcpy(buf, &value, nbytes);

    for(int i = 0; i < nbytes; ++i)
    {
        std::uint8_t cur = buf[i];
        int offset = i * CHAR_BIT;

        for(int bit = 0; bit < CHAR_BIT; ++bit)
        {
            b[offset] = cur & 1;
            ++offset;   // Move to next bit in b
            cur >>= 1;  // Move to next bit in array
        }
    }

    return b.to_string();
}
*/ 

} // namespace WithMemCpy

} // namespace Conversions
} // namespace Utilities

#endif // UTILITIES_FLOATING_POINT_H