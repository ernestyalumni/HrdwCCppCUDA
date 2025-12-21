//------------------------------------------------------------------------------
// \file BooleanAlgebra_tests.cpp
// \ref https://www.cs.utexas.edu/users/fussell/courses/cs429h/lectures/Lecture_2-429h.pdf
//------------------------------------------------------------------------------
#include "Cpp/Utilities/SuperBitSet.h"

#include <boost/test/unit_test.hpp>
#include <cmath>
#include <cstdint>

using Cpp::Utilities::SuperBitSet;

BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(BooleanAlgebra_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateBitwiseOr)
{
  // cf. https://www.cs.utexas.edu/users/fussell/courses/cs429h/lectures/Lecture_2-429h.pdf
  // pp. 16
  // A | B = 1 when either A = 1 or B = 1
  {
    SuperBitSet<2> bits {"01"};
    BOOST_TEST_REQUIRE(bits.to_string() == "01");
    BOOST_TEST_REQUIRE(bits.to_ulong() == 1);
    SuperBitSet<2> rhs_0 {0};
    BOOST_TEST_REQUIRE(rhs_0.to_string() == "00");
    BOOST_TEST_REQUIRE(rhs_0.to_ulong() == 0);   

    // 0 1 |
    // 0 0 =
    // 0 1
    BOOST_TEST((bits | rhs_0) == bits);
    BOOST_TEST((bits | rhs_0).to_string() == "01");
    SuperBitSet<2> rhs_1 {3};
    BOOST_TEST_REQUIRE(rhs_1.to_string() == "11");
    BOOST_TEST_REQUIRE(rhs_1.to_ulong() == 3);   

    // 0 1 |
    // 1 1 =
    // 1 1
    BOOST_TEST((bits | rhs_1) == rhs_1);
    BOOST_TEST((bits | rhs_1).to_string() == "11");
  }
  // Further example:
  {
    SuperBitSet<4> bits {"0101"};
    BOOST_TEST_REQUIRE(bits.to_string() == "0101");
    BOOST_TEST_REQUIRE(bits.to_ulong() == 5);
    SuperBitSet<4> rhs {3};
    BOOST_TEST_REQUIRE(rhs.to_string() == "0011");
    BOOST_TEST_REQUIRE(rhs.to_ulong() == 3);   
    SuperBitSet<4> expected_result {7};
    BOOST_TEST_REQUIRE(expected_result.to_string() == "0111");
    BOOST_TEST_REQUIRE(expected_result.to_ulong() == 7);   
    BOOST_TEST((bits | rhs) == expected_result);
    BOOST_TEST((bits | rhs).to_string() == "0111");
  }

  // cf. https://en.wikipedia.org/wiki/Bitwise_operation OR
  // bitwise OR may be used to set to 1 selected bits of register described
  // above. e.g. fourth bit of 0010 may be set by performing bitwise OR with
  // pattern with only 4th bit set.
  {
    SuperBitSet<4> bits {"0010"};
    BOOST_TEST_REQUIRE(bits.to_string() == "0010");
    BOOST_TEST_REQUIRE(bits.to_ulong() == 2);
    SuperBitSet<4> rhs {8};
    BOOST_TEST_REQUIRE(rhs.to_string() == "1000");
    BOOST_TEST_REQUIRE(rhs.to_ulong() == 8);   
    SuperBitSet<4> expected_result {10};
    BOOST_TEST_REQUIRE(expected_result.to_string() == "1010");
    BOOST_TEST_REQUIRE(expected_result.to_ulong() == 10);

    // 0 0 1 0 |
    // 1 0 0 0 =
    // 1 0 1 0
    BOOST_TEST((bits | rhs) == expected_result);
    BOOST_TEST((bits | rhs).to_string() == "1010");
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateBitwiseAnd)
{
  // cf. https://www.cs.utexas.edu/users/fussell/courses/cs429h/lectures/Lecture_2-429h.pdf
  // pp. 16
  // A & B = 1 when both A = 1 and B = 1
  {
    SuperBitSet<2> bits {"01"};
    BOOST_TEST_REQUIRE(bits.to_string() == "01");
    BOOST_TEST_REQUIRE(bits.to_ulong() == 1);
    SuperBitSet<2> rhs_0 {0};
    BOOST_TEST_REQUIRE(rhs_0.to_string() == "00");
    BOOST_TEST_REQUIRE(rhs_0.to_ulong() == 0);   
    BOOST_TEST((bits & rhs_0) == rhs_0);
    BOOST_TEST((bits & rhs_0).to_string() == "00");
    SuperBitSet<2> rhs_1 {3};
    BOOST_TEST_REQUIRE(rhs_1.to_string() == "11");
    BOOST_TEST_REQUIRE(rhs_1.to_ulong() == 3);   
    BOOST_TEST((bits & rhs_1) == bits);
    BOOST_TEST((bits & rhs_1).to_string() == "01");
  }
  // Further example:
  {
    SuperBitSet<4> bits {"0101"};
    BOOST_TEST_REQUIRE(bits.to_string() == "0101");
    BOOST_TEST_REQUIRE(bits.to_ulong() == 5);
    SuperBitSet<4> rhs {3};
    BOOST_TEST_REQUIRE(rhs.to_string() == "0011");
    BOOST_TEST_REQUIRE(rhs.to_ulong() == 3);   
    SuperBitSet<4> expected_result {1};
    BOOST_TEST_REQUIRE(expected_result.to_string() == "0001");
    BOOST_TEST_REQUIRE(expected_result.to_ulong() == 1);   

    // Only if "both" bits are 1, the resulting bit is 1.
    // 0 1 0 1 &
    // 0 0 1 1 =
    // 0 0 0 1
    BOOST_TEST((bits & rhs) == expected_result);
    BOOST_TEST((bits & rhs).to_string() == "0001");
  }

  // cf. https://en.wikipedia.org/wiki/Bitwise_operation AND
  // Thus, if both bits in compared position are 1, bit in resulting binary
  // representation is (1 x 1 = 1); otherwise result is 0 (1 x 0 = 0 and
  // 0 x 0 = 0).
  // Operation may be used to determine whether particular bit is set (1) or
  // clear (0). 
  // This is often called bit masking (by analogy, use of masking tape covers,
  // or masks, portions that are not of interest)
  {
    SuperBitSet<4> bits {"0011"};
    BOOST_TEST_REQUIRE(bits.to_string() == "0011");
    BOOST_TEST_REQUIRE(bits.to_ulong() == 3);
    SuperBitSet<4> rhs {2};
    BOOST_TEST_REQUIRE(rhs.to_string() == "0010");
    BOOST_TEST_REQUIRE(rhs.to_ulong() == 2);   
    SuperBitSet<4> expected_result {2};
    BOOST_TEST_REQUIRE(expected_result.to_string() == "0010");
    BOOST_TEST_REQUIRE(expected_result.to_ulong() == 2);
    BOOST_TEST((bits & rhs) == expected_result);
    BOOST_TEST((bits & rhs).to_string() == "0010");

    BOOST_TEST(!(bits & rhs).all());
    BOOST_TEST((bits & rhs).any());
    BOOST_TEST(!(bits & rhs).none());
  }
  // bitwise AND may be used to clear selected bits (or flags) of a register in
  // which each bit represents an individual Boolean state.
  // This technique is an efficient way to store a number of Boolean values
  // using as little memory as possible.
  {
    SuperBitSet<4> bits {"0110"};
    BOOST_TEST_REQUIRE(bits.to_string() == "0110");
    BOOST_TEST_REQUIRE(bits.to_ulong() == 6);

    // 3rd. flag may be cleared with pattern that has a 0 only in 3rd. bit.
    SuperBitSet<4> rhs {11};
    BOOST_TEST_REQUIRE(rhs.to_string() == "1011");
    BOOST_TEST_REQUIRE(rhs.to_ulong() == 11);
    SuperBitSet<4> expected_result {2};
    BOOST_TEST_REQUIRE(expected_result.to_string() == "0010");
    BOOST_TEST_REQUIRE(expected_result.to_ulong() == 2);
    BOOST_TEST((bits & rhs) == expected_result);
    BOOST_TEST((bits & rhs).to_string() == "0010");

    BOOST_TEST(!(bits & rhs).all());
    BOOST_TEST((bits & rhs).any());
    BOOST_TEST(!(bits & rhs).none());
  }

  // Also, easy to check parity (even or odd?) of binary number by checking
  // value of lowest valued bit.

  {
    SuperBitSet<4> bits {"0110"};
    BOOST_TEST_REQUIRE(bits.to_string() == "0110");
    BOOST_TEST_REQUIRE(bits.to_ulong() == 6);
    SuperBitSet<4> rhs {0001};
    BOOST_TEST_REQUIRE(rhs.to_string() == "0001");
    BOOST_TEST_REQUIRE(rhs.to_ulong() == 1);
    SuperBitSet<4> expected_result {0};
    BOOST_TEST_REQUIRE(expected_result.to_string() == "0000");
    BOOST_TEST_REQUIRE(expected_result.to_ulong() == 0);
    BOOST_TEST((bits & rhs) == expected_result);
    BOOST_TEST((bits & rhs).to_string() == "0000");

    BOOST_TEST(!(bits & rhs).all());
    BOOST_TEST(!(bits & rhs).any());
    BOOST_TEST((bits & rhs).none());
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateBitwiseXor)
{
  // cf. https://www.cs.utexas.edu/users/fussell/courses/cs429h/lectures/Lecture_2-429h.pdf
  // pp. 16
  // A ^ B = 1 when either A = 1 or B = 1, but not both
  {
    SuperBitSet<2> bits {"01"};
    BOOST_TEST_REQUIRE(bits.to_string() == "01");
    BOOST_TEST_REQUIRE(bits.to_ulong() == 1);
    SuperBitSet<2> rhs_0 {0};
    BOOST_TEST_REQUIRE(rhs_0.to_string() == "00");
    BOOST_TEST_REQUIRE(rhs_0.to_ulong() == 0);   
    BOOST_TEST((bits ^ rhs_0) == bits);
    BOOST_TEST((bits ^ rhs_0).to_string() == "01");
    SuperBitSet<2> rhs_1 {3};
    BOOST_TEST_REQUIRE(rhs_1.to_string() == "11");
    BOOST_TEST_REQUIRE(rhs_1.to_ulong() == 3);   
    BOOST_TEST((bits ^ rhs_1) == SuperBitSet<2>{"10"});
    BOOST_TEST((bits ^ rhs_1).to_string() == "10");
  }
  // Further example:
  {
    SuperBitSet<4> bits {"0101"};
    BOOST_TEST_REQUIRE(bits.to_string() == "0101");
    BOOST_TEST_REQUIRE(bits.to_ulong() == 5);
    SuperBitSet<4> rhs {3};
    BOOST_TEST_REQUIRE(rhs.to_string() == "0011");
    BOOST_TEST_REQUIRE(rhs.to_ulong() == 3);
    SuperBitSet<4> expected_result {6};
    BOOST_TEST_REQUIRE(expected_result.to_string() == "0110");
    BOOST_TEST_REQUIRE(expected_result.to_ulong() == 6);

    // 0 1 0 1 ^
    // 0 0 1 1 =
    // 0 1 1 0

    BOOST_TEST((bits ^ rhs) == expected_result);
    BOOST_TEST((bits ^ rhs).to_string() == "0110");
  }

  // cf. https://en.wikipedia.org/wiki/Bitwise_operation XOR
  // bitwise XOR may be used to invert selected bits in a register (also called
  // toggle or flip). Any bit may be toggled by XORing with 1. 
  {
    SuperBitSet<4> bits {"0010"};
    BOOST_TEST_REQUIRE(bits.to_string() == "0010");
    BOOST_TEST_REQUIRE(bits.to_ulong() == 2);
    SuperBitSet<4> rhs {10};
    BOOST_TEST_REQUIRE(rhs.to_string() == "1010");
    BOOST_TEST_REQUIRE(rhs.to_ulong() == 10);   
    SuperBitSet<4> expected_result {8};
    BOOST_TEST_REQUIRE(expected_result.to_string() == "1000");
    BOOST_TEST_REQUIRE(expected_result.to_ulong() == 8);
    BOOST_TEST((bits ^ rhs) == expected_result);
    BOOST_TEST((bits ^ rhs).to_string() == "1000");
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// NOT is also called the complement, or negation
BOOST_AUTO_TEST_CASE(DemonstrateBitwiseNot)
{
  {
    SuperBitSet<1> bits0 {"0"};
    BOOST_TEST_REQUIRE(bits0.to_string() == "0");
    BOOST_TEST((~bits0).to_string() == "1");

    SuperBitSet<1> bits1 {"1"};
    BOOST_TEST_REQUIRE(bits1.to_string() == "1");
    BOOST_TEST((~bits1).to_string() == "0");
  }

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

  {
    // cf. https://stackoverflow.com/questions/5040920/converting-from-signed-char-to-unsigned-char-and-back-again
    SuperBitSet<8> bits {static_cast<uint8_t>(static_cast<signed char>(-7))};
    BOOST_TEST(bits.to_string() == "11111001");
    BOOST_TEST(pow(2, 7) == 128);
    BOOST_TEST(pow(2, 6) == 64);
    BOOST_TEST(pow(2, 5) == 32);
    BOOST_TEST(pow(2, 4) == 16);
    BOOST_TEST(pow(2, 3) == 8);

    BOOST_TEST(
      pow(2, 0) + pow(2, 3) + pow(2, 4) + pow(2, 5) + pow(2, 6) == 121);
    BOOST_TEST(
      -pow(2,7) + pow(2, 0) + pow(2, 3) + pow(2, 4) + pow(2, 5) + pow(2, 6) ==
        -7);

    BOOST_TEST((~bits).to_string() == "00000110");
    BOOST_TEST((~bits).to_ulong() == 6);
  }
  {
    SuperBitSet<8> bits {0xff};
    BOOST_TEST(bits.to_string() == "11111111");
    BOOST_TEST(bits.to_ulong() == 255);
    BOOST_TEST((~bits).to_string() == "00000000");
    BOOST_TEST((~bits).to_ulong() == 0);
  }
  {
    SuperBitSet<8> bits {static_cast<uint8_t>(static_cast<signed char>(-128))};
    BOOST_TEST(bits.to_string() == "10000000");
    BOOST_TEST(bits.to_ulong() == 128);

    BOOST_TEST((~bits).to_string() == "01111111");
    BOOST_TEST((~bits).to_ulong() == 127);
  }
  {
    SuperBitSet<8> bits {static_cast<uint8_t>(static_cast<signed char>(-1))};
    BOOST_TEST(bits.to_string() == "11111111");
    BOOST_TEST(bits.to_ulong() == 255);

    BOOST_TEST((~bits).to_string() == "00000000");
    BOOST_TEST((~bits).to_ulong() == 0);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateTwosComplement)
{

}

BOOST_AUTO_TEST_SUITE_END() // BooleanAlgebra_tests
BOOST_AUTO_TEST_SUITE_END() // Utilities