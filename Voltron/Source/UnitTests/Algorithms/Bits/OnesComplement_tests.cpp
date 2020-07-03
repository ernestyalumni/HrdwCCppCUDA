//------------------------------------------------------------------------------
/// \file OnesComplement_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief Demonstrating One's Complement and One's Complement Sum.
/// \details Span is a class template and belongs in Containers library.
/// \ref https://en.cppreference.com/w/cpp/container/span
//------------------------------------------------------------------------------
#include "Algorithms/Bits/Masks.h"
#include "Cpp/Numerics/BitCast.h"
#include "Cpp/Utilities/SuperBitSet.h"
#include "Utilities/ToHexString.h"

#include <boost/test/unit_test.hpp>
#include <iostream>

using Algorithms::Bits::clear_bit;
using Cpp::Numerics::bit_cast;
using Cpp::Utilities::SuperBitSet;
using Utilities::ToHexString;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(Bits)
BOOST_AUTO_TEST_SUITE(OnesComplement_tests)


// cf. https://en.wikipedia.org/wiki/Ones%27_complement
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(OnesComplementBinaryRepresentation)
{
  {
    // From bits
    const uint8_t x {0b01111111};
    BOOST_TEST(x == 127);

    SuperBitSet<8> xbits8 {x};
    BOOST_TEST(xbits8.to_string() == "01111111");

    ToHexString xhex {x};
    BOOST_TEST(xhex.as_increasing_addresses() == "7f");

    BOOST_TEST(~x == -128);

    BOOST_TEST((~xbits8).to_string() == "10000000");
    BOOST_TEST((~xbits8).to_ulong() == 128);

    const uint8_t uxc {0b10000000};
    BOOST_TEST(uxc == 128);

    const int8_t y {-128};
    SuperBitSet<8> ybits8 {bit_cast<uint8_t>(y)};
    BOOST_TEST(ybits8.to_string() == "10000000");

    ToHexString yhex {y};
    BOOST_TEST(yhex.as_increasing_addresses() == "80");


//    SuperBitSet<8> xcbits8 {~x};
//    BOOST_TEST(xcbits8.to_string() == "10000000");

  } 


}

// https://stackoverflow.com/questions/5607978/how-is-a-1s-complement-checksum-useful-for-error-detection
// https://stackoverflow.com/posts/41247351/revisions
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StackoverflowOnesComplementSumExample)
{
  const std::size_t word_size {sizeof(uint16_t)};

  // Example, you have 3 words in UDP packet needed to be sent:

  constexpr uint16_t word_1 {0b0110011001100000};
  constexpr uint16_t word_2 {0b0101010101010101};
  constexpr uint16_t word_3 {0b1000111100001100};

  constexpr uint16_t max_limit {0b1111111111111111};

  SuperBitSet<16> word_1bits16 {word_1};
  SuperBitSet<16> word_2bits16 {word_2};
  SuperBitSet<16> word_3bits16 {word_3};

  SuperBitSet<16> max_limitbits16 {max_limit};

  BOOST_TEST(word_1bits16.to_ulong() == 26208);
  BOOST_TEST(word_2bits16.to_ulong() == 21845);
  BOOST_TEST(word_3bits16.to_ulong() == 36620);

  BOOST_TEST(max_limitbits16.to_ulong() == 65535);

  constexpr uint16_t word_12 {word_1 + word_2};
  SuperBitSet<16> word_12bits16 {word_12};
  BOOST_TEST(word_12bits16.to_string() == "1011101110110101");
  BOOST_TEST(word_12bits16.to_ulong() == 48053);

  // In order to do the one's complement sum, use the trick of a larger sized
  // type to contain the carried over bit in the most-significant position, and
  // check if the resulting sum is greater than 
  const uint32_t word_123 {word_12 + word_3};
  SuperBitSet<32> word_123bits32 {word_123};
  BOOST_TEST(word_123bits32.to_string() == "00000000000000010100101011000001");

  const uint16_t reduced_word_123 {
    static_cast<uint16_t>(clear_bit(word_123, 16))};
  SuperBitSet<16> reduced_word_123bits16 {reduced_word_123};
  BOOST_TEST(reduced_word_123bits16.to_string() == "0100101011000001");

  BOOST_TEST(reduced_word_123 < word_3);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(UnsignedIntegerOverflowFor64Bit)
{
  uint64_t sum {0xffffffffffffffff};

  SuperBitSet<64> sumbits64 {sum};
  BOOST_TEST(
    sumbits64.to_string() ==
      "1111111111111111111111111111111111111111111111111111111111111111");

  constexpr uint64_t operand {0x18};
  SuperBitSet<64> operandbits64 {operand};
  BOOST_TEST(
    operandbits64.to_string() ==
      "0000000000000000000000000000000000000000000000000000000000011000");

  sum += operand;

  BOOST_TEST(sizeof(sum) == sizeof(uint64_t));

  SuperBitSet<64> postsumbits64 {sum};
  BOOST_TEST(
    postsumbits64.to_string() ==
      "0000000000000000000000000000000000000000000000000000000000010111");

}

BOOST_AUTO_TEST_SUITE_END() // OnesComplement_tests
BOOST_AUTO_TEST_SUITE_END() // Bits
BOOST_AUTO_TEST_SUITE_END() // Algorithms