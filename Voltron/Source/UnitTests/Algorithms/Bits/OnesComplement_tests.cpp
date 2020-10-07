//------------------------------------------------------------------------------
/// \file OnesComplement_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief Demonstrating One's Complement and One's Complement Sum.
/// \details Span is a class template and belongs in Containers library.
/// \ref https://en.cppreference.com/w/cpp/container/span
//------------------------------------------------------------------------------
#include "Algorithms/Bits/Masks.h"
#include "Algorithms/Bits/OnesComplementSum.h"
#include "Cpp/Numerics/BitCast.h"
#include "Cpp/Utilities/SuperBitSet.h"
#include "Utilities/ToHexString.h"

#include <boost/test/unit_test.hpp>
#include <cstdlib>
#include <iostream>
#include <vector>

using Algorithms::Bits::ones_complement_binary_sum;
using Algorithms::Bits::ones_complement_sum;
using Algorithms::Bits::clear_bit;
using Cpp::Numerics::bit_cast;
using Cpp::Utilities::SuperBitSet;
using Utilities::ToHexString;
using std::uint16_t;

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
    // *** stack smashing detected ***: terminated
    // unknown location(0): fatal error: in "Algorithms/Bits/OnesComplement_tests/OnesComplementBinaryRepresentation": signal: SIGABRT (application abort requested)

    //BOOST_TEST(xhex.as_increasing_addresses() == "7f");

    BOOST_TEST(~x == -128);

    BOOST_TEST((~xbits8).to_string() == "10000000");
    BOOST_TEST((~xbits8).to_ulong() == 128);

    const uint8_t uxc {0b10000000};
    BOOST_TEST(uxc == 128);

    const int8_t y {-128};
    SuperBitSet<8> ybits8 {bit_cast<uint8_t>(y)};
    BOOST_TEST(ybits8.to_string() == "10000000");

    ToHexString yhex {y};

    // *** stack smashing detected ***: terminated
    // unknown location(0): fatal error: in "Algorithms/Bits/NumericalRepresentation_tests/TwosComplementBinaryAdditionWhenSumIsNotArithemticallyCorrect": signal: SIGABRT (application abort requested) 
    //BOOST_TEST(yhex.as_increasing_addresses() == "80");

//    SuperBitSet<8> xcbits8 {~x};
//    BOOST_TEST(xcbits8.to_string() == "10000000");

  } 
}

constexpr uint16_t max_limit_as_hex {0xffff};

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
  SuperBitSet<16> max_limit_as_hex_bits16 {max_limit_as_hex};

  BOOST_TEST(word_1bits16.to_ulong() == 26208);
  BOOST_TEST(word_2bits16.to_ulong() == 21845);
  BOOST_TEST(word_3bits16.to_ulong() == 36620);

  BOOST_TEST(max_limitbits16.to_ulong() == 65535);
  BOOST_TEST(max_limit_as_hex_bits16.to_ulong() == 65535);
  BOOST_TEST(max_limit_as_hex_bits16.to_string() == "1111111111111111");

  constexpr uint16_t word_12 {word_1 + word_2};
  SuperBitSet<16> word_12bits16 {word_12};
  BOOST_TEST(word_12bits16.to_string() == "1011101110110101");
  BOOST_TEST(word_12bits16.to_ulong() == 48053);
  BOOST_TEST(word_12 < max_limit_as_hex);

  // In order to do the one's complement sum, use the trick of a larger sized
  // type to contain the carried over bit in the most-significant position, and
  // check if the resulting sum is greater than 
  const uint32_t word_123 {word_12 + word_3};
  BOOST_TEST(word_123 > max_limit_as_hex);
  SuperBitSet<32> word_123bits32 {word_123};
  BOOST_TEST(word_123bits32.to_string() == "00000000000000010100101011000001");

  const uint16_t reduced_word_123 {
    static_cast<uint16_t>(clear_bit(word_123, 16))};
  SuperBitSet<16> reduced_word_123bits16 {reduced_word_123};
  BOOST_TEST(reduced_word_123bits16.to_string() == "0100101011000001");

  BOOST_TEST(reduced_word_123 < word_3);
}

// cf. http://mathforum.org/library/drmath/view/54379.html
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ProducingCarryOutInUint16_t)
{
  constexpr uint16_t word_1 {0b1000011001011110};
  constexpr uint16_t word_2 {0b1010110001100000};
  constexpr uint16_t word_3 {0b0111000100101010};
  constexpr uint16_t word_4 {0b1000000110110101};

  auto word_12 {word_1 + word_2};
  BOOST_TEST(word_12 > max_limit_as_hex);

  // Operator precedence for > over += assignment.
  // Do the carry first.
  word_12 += word_12 > max_limit_as_hex;

  uint16_t carried_word_12 {
    // Recall that clear_bit is the mask (x & ~(1 << position))
    static_cast<uint16_t>(clear_bit(word_12, 16))};
  SuperBitSet<16> carried_word_12bits16 {carried_word_12};
  BOOST_TEST(carried_word_12bits16.to_string() == "0011001010111111");

  auto word_123 {carried_word_12 + word_3};
  BOOST_TEST(word_123 <= max_limit_as_hex);
  SuperBitSet<16> word_123bits16 {static_cast<uint16_t>(word_123)};
  BOOST_TEST(word_123bits16.to_string() == "1010001111101001");

  auto word_1234 {word_123 + word_4};
  BOOST_TEST(word_1234 > max_limit_as_hex);

  word_1234 += word_1234 > max_limit_as_hex;
  uint16_t carried_word_1234 {
    static_cast<uint16_t>(clear_bit(word_1234, 16))};
  SuperBitSet<16> carried_word_1234bits16 {carried_word_1234};
  BOOST_TEST(carried_word_1234bits16.to_string() == "0010010110011111");

  const uint16_t ones_complement_sum_result {
    static_cast<uint16_t>(~carried_word_1234)};
  SuperBitSet<16> ones_complement_sum_resultbits16 {
    ones_complement_sum_result};
  BOOST_TEST(ones_complement_sum_resultbits16.to_string() ==
    "1101101001100000");
  // 0010010110011111
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

BOOST_AUTO_TEST_SUITE(OnesComplementSum_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(OnesComplementSumOfUint16_t)
{
  constexpr uint16_t word_1 {0b1000011001011110};
  constexpr uint16_t word_2 {0b1010110001100000};
  constexpr uint16_t word_3 {0b0111000100101010};
  constexpr uint16_t word_4 {0b1000000110110101};

  std::vector<uint16_t> values;
  values.emplace_back(word_1);
  values.emplace_back(word_2);
  values.emplace_back(word_3);
  values.emplace_back(word_4);

  const uint16_t sum_result {ones_complement_sum(values)};
  SuperBitSet<16> sum_resultbits16 {sum_result};
  BOOST_TEST(sum_resultbits16.to_string() == "1101101001100000");
}

// cf. http://kfall.net/ucbpage/EE122/lec06/lec06-outline.pdf
// http://kfall.net/ucbpage/EE122/lec06/sld023.htm
// http://kfall.net/ucbpage/EE122/lec06/tsld022.htm
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(OnesComplementSumExample)
{
  constexpr uint16_t word_1 {0xe34f};
  constexpr uint16_t word_2 {0x2396};
  constexpr uint16_t word_3 {0x4427};
  constexpr uint16_t word_4 {0x99f3};

  const uint32_t twos_complement_sum {word_1 + word_2 + word_3 + word_4};

  BOOST_TEST(twos_complement_sum == 0x01e4ff);

  std::vector<uint16_t> values;
  values.emplace_back(word_1);
  values.emplace_back(word_2);
  values.emplace_back(word_3);
  values.emplace_back(word_4);

  const uint16_t sum_result {ones_complement_sum(values)};
  SuperBitSet<16> sum_resultbits16 {sum_result};
  BOOST_TEST(sum_resultbits16.to_ulong() == 0x1aff);
}

BOOST_AUTO_TEST_SUITE_END() // OnesComplementSum_tests

BOOST_AUTO_TEST_SUITE_END() // OnesComplement_tests
BOOST_AUTO_TEST_SUITE_END() // Bits
BOOST_AUTO_TEST_SUITE_END() // Algorithms