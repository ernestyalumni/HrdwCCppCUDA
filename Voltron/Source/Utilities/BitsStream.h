//------------------------------------------------------------------------------
/// \file BitsStream.h
/// \author
/// \brief .
/// \ref 
///-----------------------------------------------------------------------------
#ifndef UTILITIES_BITS_STREAM_H
#define UTILITIES_BITS_STREAM_H

#include <cstddef> // std::size_t, std::byte
#include <cstdio> // printf
#include <string>

namespace Utilities
{

class BitsStream
{
  public:

    // TODO: Understand 1's complement arithmetic and how this is equal to -0
    static constexpr uint16_t minus_0_1s_complement = 0xffffu;

    BitsStream(
      std::byte* const data,
      const std::size_t N_bytes,
      const std::size_t initialized_bits = 0);

    ~BitsStream() = default;

  private:

    // Raw pointer (no specific ownership) to data buffer.
    // cf. https://en.cppreference.com/w/cpp/types/byte
    // std::byte is distinct type that implements concept of byte as specified
    // in C++ language definition.
    // Like char and unsigned char, it can be used to access raw memory occupied
    // by other objects (object representation), but unlike those types, it's
    // not a character type and isn't an arithmetic type.
    // A byte is only a collection of bits, and only operators defined for it
    // are bitwise ones.
    std::byte* const data_;
};

} // namespace Utilities

#endif // UTILITIES_BITS_STREAM_H