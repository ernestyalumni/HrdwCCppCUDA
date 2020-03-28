//------------------------------------------------------------------------------
// \file BitSet_test.cpp
//------------------------------------------------------------------------------
#include "Cpp/Utilities/SuperBitSet.h"

#include <bitset>
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
// https://stackoverflow.com/questions/20731/how-do-you-clear-a-stringstream-variable
#include <sstream>

using Cpp::Utilities::SuperBitSet;
using Cpp::Utilities::number_of_bits_in_a_byte;

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(SuperBitSet_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(NumberOfBitsInAByteIs8)
{
  BOOST_TEST(number_of_bits_in_a_byte == 8);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BitSetWorks)
// cf. https://www.cs.utexas.edu/users/fussell/courses/cs429h/lectures/Lecture_3-429h.pdf
{
  BOOST_TEST_REQUIRE(sizeof(unsigned short) == 2); // 2 bytes long
  BOOST_TEST_REQUIRE(std::pow(2, 2*8) == 65536); // 2 bytes long, 8 bits in a byte

  std::bitset<sizeof(unsigned short) * 8> unsigned_short_bitset {15213};

  constexpr unsigned short x_0 {15213};
  constexpr short x_1 {15213};
  constexpr short y {-15213};

  //std::cout << unsigned_short_bitset.to_string() << '\n';
  BOOST_TEST(unsigned_short_bitset.to_string() == "0011101101101101");
  unsigned_short_bitset = x_0;
  BOOST_TEST(unsigned_short_bitset.to_string() == "0011101101101101");
  unsigned_short_bitset = x_1;
  BOOST_TEST(unsigned_short_bitset.to_string() == "0011101101101101");
  unsigned_short_bitset = y;
  BOOST_TEST(unsigned_short_bitset.to_string() == "1100010010010011");
}

// cf. https://www.geeksforgeeks.org/c-bitset-and-its-application/
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateStdBitSet)
{
  {
    // Setup
    std::stringstream string_stream;

    constexpr std::size_t M {32};
    // default constructor initializes with all bits 0
    std::bitset<M> bset1;

    // bset2 is initialized with bits of 20
    std::bitset<M> bset2 {20};

    // bset3 is initialized with bits of specified binary string
    std::bitset<M> bset3 {std::string{"1100"}};

    // std::cout prints exact bits representation of bitset
    std::cout << bset1 << std::endl; // 00000000000000000000000000000000
    std::cout << bset2 << std::endl; // 00000000000000000000000000010100
    std::cout << bset3 << std::endl; // 00000000000000000000000000001100

    // declaring set8 with capacity of 8 bits
    std::bitset<8> set8; // 00000000

    // setting first bit (or 6th index)
    set8[1] = 1; // 00000010
    set8[4] = set8[1]; // 00010010
    std::cout << set8 << std::endl;

    // count function returns number of set bits in bitset
    int numberof1 = set8.count();
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateSuperBitSet)
{
  {
    SuperBitSet<2> bits2 {1};
    BOOST_TEST(bits2.to_string() == "01");
  }

  BOOST_TEST(true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SuperBitSetIsLittleEndian)
{
  {
    // initializing 1st M=5 (rightmost, least significant) bit position
    SuperBitSet<5> bits {"11010"};
    BOOST_TEST(bits.to_string() == "11010");
    BOOST_TEST(bits[0] == 0);
    BOOST_TEST(bits[1] == 1);
    BOOST_TEST(bits[2] == 0);
    BOOST_TEST(bits[3] == 1);
    BOOST_TEST(bits[4] == 1);
  }
  {
    SuperBitSet<8> bits {0};
    BOOST_TEST(bits[0] == 0);
    BOOST_TEST(bits[1] == 0);
    BOOST_TEST(bits[2] == 0);
    BOOST_TEST(bits[3] == 0);
    BOOST_TEST(bits[4] == 0);
    BOOST_TEST(bits[5] == 0);
    BOOST_TEST(bits[6] == 0);
    BOOST_TEST(bits[7] == 0);
  }
  {
    SuperBitSet<8> bits {1};
    BOOST_TEST(bits[0] == 1);
    BOOST_TEST(bits[1] == 0);
    BOOST_TEST(bits[2] == 0);
    BOOST_TEST(bits[3] == 0);
    BOOST_TEST(bits[4] == 0);
    BOOST_TEST(bits[5] == 0);
    BOOST_TEST(bits[6] == 0);
    BOOST_TEST(bits[7] == 0);
  }
  {
    SuperBitSet<8> bits {2};
    BOOST_TEST(bits[0] == 0);
    BOOST_TEST(bits[1] == 1);
    BOOST_TEST(bits[2] == 0);
    BOOST_TEST(bits[3] == 0);
    BOOST_TEST(bits[4] == 0);
    BOOST_TEST(bits[5] == 0);
    BOOST_TEST(bits[6] == 0);
    BOOST_TEST(bits[7] == 0);
  }
  {
    SuperBitSet<8> bits {std::numeric_limits<uint8_t>::max() - 1};
    BOOST_TEST(bits[0] == 0);
    BOOST_TEST(bits[1] == 1);
    BOOST_TEST(bits[2] == 1);
    BOOST_TEST(bits[3] == 1);
    BOOST_TEST(bits[4] == 1);
    BOOST_TEST(bits[5] == 1);
    BOOST_TEST(bits[6] == 1);
    BOOST_TEST(bits[7] == 1);
  }
  {
    SuperBitSet<8> bits {std::numeric_limits<uint8_t>::max()};
    BOOST_TEST(bits[0] == 1);
    BOOST_TEST(bits[1] == 1);
    BOOST_TEST(bits[2] == 1);
    BOOST_TEST(bits[3] == 1);
    BOOST_TEST(bits[4] == 1);
    BOOST_TEST(bits[5] == 1);
    BOOST_TEST(bits[6] == 1);
    BOOST_TEST(bits[7] == 1);
  }
  {
    SuperBitSet<8> bits {std::numeric_limits<uint8_t>::max() + 1};
    BOOST_TEST(bits[0] == 0);
    BOOST_TEST(bits[1] == 0);
    BOOST_TEST(bits[2] == 0);
    BOOST_TEST(bits[3] == 0);
    BOOST_TEST(bits[4] == 0);
    BOOST_TEST(bits[5] == 0);
    BOOST_TEST(bits[6] == 0);
    BOOST_TEST(bits[7] == 0);
  }

  BOOST_TEST(true);
}

// Use examples from wikipedia, "Bitwise operation"

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// NOT is also called the complement, or negation
BOOST_AUTO_TEST_CASE(SuperBitSetWorksWithBitwiseNot)
{
  // bitwise complement = two's complement of the value - 1. If 2's complement
  // arithmetic used, then NOT x = -x - 1
  // TODO: explore and understand the previous statement.
  {
    SuperBitSet<4> bits {7};
    BOOST_TEST(bits.to_string() == "0111");
    BOOST_TEST(bits.to_ulong() == 7);
    BOOST_TEST((~bits).to_string() == "1000");    
    BOOST_TEST((~bits).to_ulong() == 8);
  }
  {
    SuperBitSet<8> bits {171};
    BOOST_TEST(bits.to_string() == "10101011");
    BOOST_TEST(bits.to_ulong() == 171);
    BOOST_TEST((~bits).to_string() == "01010100");    
    BOOST_TEST((~bits).to_ulong() == 84);    
  }
  // For unsigned ints, bitwise complement of number is "mirror reflection" of
  // number across half-way point of unsigned int's range;
  // e.g. for 8-bit unsigned int, NOT x = 255 - x, "flips" increasing range from
  // 0 to 255, to decreasing range from 255 to 0.
}

// cf. https://en.wikipedia.org/wiki/Bitwise_operation Arithmetic shift section.

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SuperBitSetWorksWithLeftShift)
{
  {
    SuperBitSet<8> bits {"00010111"};
    BOOST_TEST_REQUIRE(bits.to_string() == "00010111");
    BOOST_TEST(bits.to_ulong() == 23);
    BOOST_TEST((bits << 1).to_string() == "00101110");
    BOOST_TEST((bits << 1).to_ulong() == 46);
    BOOST_TEST((bits << 2).to_string() == "01011100");
    BOOST_TEST((bits << 2).to_ulong() == 92);
  }
}

// cf. https://en.wikipedia.org/wiki/Bitwise_operation Arithmetic shift section.
// In a right arithmetic shift, sign bit (MSB in two's complement) is shifted in
// on left, preserves sign of operand. If MSB is 1, new 1 is copied into
// leftmost position.
// Logical right-shift inserts 0 into most significant bit, instead of copying
// sign bit; it's ideal for unsigned binary numbers, while
// arithmetic right-shift ideal for signed two's complement binary numbers.
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SuperBitSetWorksWithRightShiftAsLogicalShift)
{
  {
    SuperBitSet<8> bits {"10010111"};
    BOOST_TEST_REQUIRE(bits.to_string() == "10010111");
    BOOST_TEST(bits.to_ulong() == 151);
    BOOST_TEST((bits >> 1).to_string() == "01001011");
    BOOST_TEST((bits >> 1).to_ulong() == 75);
    BOOST_TEST((bits >> 2).to_string() == "00100101");
    BOOST_TEST((bits >> 2).to_ulong() == 37);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(UseBitwiseLeftShiftToGetMax)
{
  // unsigned long long is 64 bits or 8 bytes.
  SuperBitSet<64> bits {1ull};
  BOOST_TEST(sizeof(unsigned long long) == 8);
  {
    BOOST_TEST_REQUIRE(
      bits.to_string() ==
        "0000000000000000000000000000000000000000000000000000000000000001");
    BOOST_TEST(bits.to_ullong() == 1ull);

    BOOST_TEST((bits << 0).to_string() ==
      "0000000000000000000000000000000000000000000000000000000000000001");
    BOOST_TEST((bits << 0).to_ullong() == 1);
    BOOST_TEST((bits << 1).to_string() ==
      "0000000000000000000000000000000000000000000000000000000000000010");
    BOOST_TEST((bits << 1).to_ullong() == 2);
    BOOST_TEST((bits << 4).to_string() ==
      "0000000000000000000000000000000000000000000000000000000000010000");
    BOOST_TEST((bits << 4).to_ullong() == 16);

    BOOST_TEST(SuperBitSet<8>{(bits << 0).to_ullong() - 1}.to_string() ==
      "00000000");
    BOOST_TEST(SuperBitSet<8>{(bits << 0).to_ullong() - 1}.to_ulong() == 0);
    BOOST_TEST(SuperBitSet<8>{(bits << 1).to_ullong() - 1}.to_string() ==
      "00000001");
    BOOST_TEST(SuperBitSet<8>{(bits << 1).to_ullong() - 1}.to_ulong() == 1);
    BOOST_TEST(SuperBitSet<8>{(bits << 4).to_ullong() - 1}.to_string() ==
      "00001111");
    BOOST_TEST(SuperBitSet<8>{(bits << 4).to_ullong() - 1}.to_ulong() == 15);
  }
}

BOOST_AUTO_TEST_SUITE_END() // SuperBitSet_tests
BOOST_AUTO_TEST_SUITE_END() // Utilities
BOOST_AUTO_TEST_SUITE_END() // Cpp