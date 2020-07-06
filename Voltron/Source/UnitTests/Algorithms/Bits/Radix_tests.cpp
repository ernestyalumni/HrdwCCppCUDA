//------------------------------------------------------------------------------
/// \file Radix_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://en.wikipedia.org/wiki/Radix
/// https://www.youtube.com/watch?v=NLKQEOgBAnw
/// Algorithms: Bit Manipulation. HackerRank. Gayle Laakmann McDowell.
/// \details To run only the Bits unit tests, do this:
/// ./Check --run_test="Algorithms/Bits"
/// 
/// Also, consider running the BooleanAlgebra_tests, as such:
/// ./Check --run_test="Utilities/BooleanAlgebra_tests"
//------------------------------------------------------------------------------
#include "Algorithms/Bits/Masks.h"
#include "Algorithms/Bits/Shift.h"
#include "Cpp/Numerics/BitCast.h"
#include "Cpp/Utilities/SuperBitSet.h"
#include "Utilities/EndianConversions.h"
#include "Utilities/ToHexString.h"

#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <iostream>
#include <limits>

using Algorithms::Bits::clear_bit;
using Algorithms::Bits::is_bit_set_high;
using Algorithms::Bits::set_bit_high;
using Algorithms::Bits::logical_right_shift;
using Cpp::Numerics::bit_cast;
using Cpp::Utilities::SuperBitSet;
using Cpp::Utilities::number_of_bits_in_a_byte;
using Utilities::ToHexString;
using Utilities::to_big_endian;
using Utilities::to_little_endian;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(Bits)
BOOST_AUTO_TEST_SUITE(NumericalRepresentation_tests)

// cf. https://www.cs.utexas.edu/users/fussell/courses/cs429h/lectures/Lecture_2-429h.pdf
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BinaryRepresentation)
{
  // Limits
  {
    std::cout << "\n\n Unsigned limits \n";

    std::cout << "std::numeric_limits<uint8_t>::max(): " <<
      static_cast<unsigned int>(std::numeric_limits<uint8_t>::max()) << "\n";
        // 255
    std::cout << "std::numeric_limits<uint16_t>::max(): " <<
      std::numeric_limits<uint16_t>::max() << "\n"; // 65535
    std::cout << "std::numeric_limits<uint32_t>::max(): " <<
      std::numeric_limits<uint32_t>::max() << "\n"; // 4294967295
    std::cout << "std::numeric_limits<uint64_t>::max(): " <<
      std::numeric_limits<uint64_t>::max() << "\n"; // 18446744073709551615

    std::cout << "\n FloatingPoint limits \n";

    std::cout << "std::numeric_limits<float>::max(): " <<
      std::numeric_limits<float>::max() << "\n";
    std::cout << "std::numeric_limits<double>::max(): " <<
      std::numeric_limits<double>::max() << "\n";
  }

  {
    std::cout << "\n\n Signed limits \n";

    std::cout << "std::numeric_limits<int8_t>::max(): " <<
      static_cast<unsigned int>(std::numeric_limits<int8_t>::max()) << "\n";
        // 127
    std::cout << "std::numeric_limits<int16_t>::max(): " <<
      std::numeric_limits<int16_t>::max() << "\n"; // 32767
    std::cout << "std::numeric_limits<int32_t>::max(): " <<
      std::numeric_limits<int32_t>::max() << "\n"; // 2147483647
    std::cout << "std::numeric_limits<int64_t>::max(): " <<
      std::numeric_limits<int64_t>::max() << "\n"; // 18446744073709551615

    std::cout << "std::numeric_limits<int8_t>::min(): " <<
      static_cast<int>(std::numeric_limits<int8_t>::min()) << "\n";
        // -128
    std::cout << "std::numeric_limits<int16_t>::min(): " <<
      std::numeric_limits<int16_t>::min() << "\n"; // -32768
    std::cout << "std::numeric_limits<int32_t>::min(): " <<
      std::numeric_limits<int32_t>::min() << "\n"; // -2147483648
    std::cout << "std::numeric_limits<int64_t>::min(): " <<
      std::numeric_limits<int64_t>::min() << "\n"; // 18446744073709551615
  }

  // Examples
  {
    std::cout << "\n Examples of Binary Representations \n";

    {
      SuperBitSet<sizeof(uint16_t) * number_of_bits_in_a_byte> x {15213};
      std::cout << "15213_10 as binary: " << x.to_string() << "\n";
    }
  }

}

// cf. https://en.cppreference.com/w/cpp/language/integer_literal
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IntegerLiteralBitsAsChars)
{
  {
    const uint8_t au {0b00};
    BOOST_TEST_REQUIRE(au == 0);

    SuperBitSet<8> bits8 {au};
    BOOST_TEST(bits8.to_string() == "00000000");

    const char ac {0b00};
    SuperBitSet<8> cbits8 {ac};
    BOOST_TEST(cbits8.to_string() == "00000000");
  } 
  {
    const uint8_t au {0b01};
    BOOST_TEST_REQUIRE(au == 1);

    SuperBitSet<8> bits8 {au};
    BOOST_TEST(bits8.to_string() == "00000001");

    const char ac {0b01};
    SuperBitSet<8> cbits8 {ac};
    BOOST_TEST(cbits8.to_string() == "00000001");
  } 
  {
    const uint8_t au {0b10};
    BOOST_TEST_REQUIRE(au == 2);

    SuperBitSet<8> bits8 {au};
    BOOST_TEST(bits8.to_string() == "00000010");

    const char ac {0b10};
    SuperBitSet<8> cbits8 {ac};
    BOOST_TEST(cbits8.to_string() == "00000010");
  } 
  {
    const uint8_t au {0b11};
    BOOST_TEST_REQUIRE(au == 3);

    SuperBitSet<8> bits8 {au};
    BOOST_TEST(bits8.to_string() == "00000011");

    const char ac {0b11};
    SuperBitSet<8> cbits8 {ac};
    BOOST_TEST(cbits8.to_string() == "00000011");
  } 
}

// cf. https://www.youtube.com/watch?v=NLKQEOgBAnw
// Algorithms: Bit Manipulation,  HackerRank with Gayle Laakmann McDowell.
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BinaryAdditionCarriesOne)
{
  const uint8_t a {0b0101};
  const uint8_t b {0b0011};
  const uint8_t c {a + b};
  SuperBitSet<8> c8 {c};
  BOOST_TEST(c8.to_string() == "00001000");
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TwosComplement)
{
  // cf. https://www.cs.utexas.edu/users/fussell/courses/cs429h/lectures/Lecture_3-429h.pdf
  // "Encoding Integers." 
  {
    const int16_t x {15213};
    const SuperBitSet<16> xbits16 {x};
    BOOST_TEST(xbits16.to_string() == "0011101101101101");
    ToHexString<int16_t> xh {x};
    BOOST_TEST(xh() == "6d3b");
    ToHexString<int16_t> be_xh {to_big_endian(xh)};
    BOOST_TEST(be_xh() == "3b6d");

    const int16_t y {-15213};
    const auto yu = bit_cast<uint16_t>(y);
    const SuperBitSet<16> yubits16 {yu};
    BOOST_TEST(yubits16.to_string() == "1100010010010011");
    ToHexString<int16_t> yh {y};
    BOOST_TEST(yh() == "93c4");
    ToHexString<int16_t> be_yh {to_big_endian(yh)};
    BOOST_TEST(be_yh() == "c493");
  }

  {
    const int8_t a {18};
    const auto au = bit_cast<uint8_t>(a);
    SuperBitSet<8> aubits8 {au};
    // Sign bit is 0 for positive.
    BOOST_TEST(aubits8.to_string() == "00010010");
  }
  {
    const int8_t a {-18};
    const auto au = bit_cast<uint8_t>(a);
    SuperBitSet<8> aubits8 {au};
    // Sign bit is 0 for positive.
    BOOST_TEST(aubits8.to_string() == "11101110");
  }
}

// cf. http://sandbox.mc.edu/~bennet/cs110/tc/add.html
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(
  TwosComplementBinaryAdditionWhenSumIsNotArithemticallyCorrect)
{
  // Overflow, no carryout (at sign bit). Sum is not correct.
  {
    const int8_t x {104};
    const SuperBitSet<8> xbits8 {x};
    BOOST_TEST(xbits8.to_string() == "01101000");
    ToHexString<int8_t> xh {x};
    BOOST_TEST(xh() == "68");

    const int8_t y {45};
    const SuperBitSet<8> ybits8 {y};
    BOOST_TEST(ybits8.to_string() == "00101101");
    ToHexString<int8_t> yh {y};
    BOOST_TEST(yh() == "2d");

    const auto z = x + y;
    BOOST_TEST(sizeof(z) == 4);
    const SuperBitSet<8> zbits8 {z};
    BOOST_TEST(zbits8.to_string() == "10010101");
    BOOST_TEST(zbits8.to_ulong() == 149);

    // "Wraps around" to -128 and "up".
    const int8_t z8 {-107};
    const SuperBitSet<8> z8bits8 {bit_cast<uint8_t>(z8)};
    BOOST_TEST(z8bits8.to_string() == "10010101");
  }
  // Overflow, with incidental carryout. Sum is not correct.
  {
    const int8_t x {-103};
    const SuperBitSet<8> xbits8 {bit_cast<uint8_t>(x)};
    BOOST_TEST(xbits8.to_string() == "10011001");
    ToHexString<int8_t> xh {x};
    BOOST_TEST(xh() == "99");

    const int8_t y {-69};
    const SuperBitSet<8> ybits8 {bit_cast<uint8_t>(y)};
    BOOST_TEST(ybits8.to_string() == "10111011");
    ToHexString<int8_t> yh {y};
    BOOST_TEST(yh() == "bb");

    const auto z = x + y;
    BOOST_TEST(sizeof(z) == 4);
    const SuperBitSet<32> zbits32 {bit_cast<uint32_t>(z)};
    BOOST_TEST(zbits32.to_string() == "11111111111111111111111101010100");
    BOOST_TEST(z == -172);

    // "Wraps around" to -128 and "up".
    const int8_t z8 {84};
    const SuperBitSet<8> z8bits8 {bit_cast<uint8_t>(z8)};
    BOOST_TEST(z8bits8.to_string() == "01010100");
  }
}

// cf. https://www.youtube.com/watch?v=NLKQEOgBAnw
// Algorithms: Bit Manipulation,  HackerRank with Gayle Laakmann McDowell.
// This test breaks down step by step the algorithm Gayle described clearly that
// also explains the name two's complement to get the additive inverse (i.e.
// the negative) of a positive integer.
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(GetAdditiveInverseFromComplement)
{
  const int8_t x {18};
  SuperBitSet<8> xbits8 {bit_cast<uint8_t>(x)};
  BOOST_TEST(xbits8.to_string() == "00010010");

  BOOST_TEST((~xbits8).to_string() == "11101101");

  const int8_t not_x {~x};
  SuperBitSet<8> not_xbits8 {bit_cast<uint8_t>(not_x)};
  BOOST_TEST(not_xbits8.to_string() == "11101101");

  const int8_t not_x_plus_1 {not_x + 1};
  SuperBitSet<8> not_x_plus_1bits8 {bit_cast<uint8_t>(not_x_plus_1)};
  BOOST_TEST(not_x_plus_1bits8.to_string() == "11101110");

  const int8_t y {-18};
  SuperBitSet<8> ybits8 {bit_cast<uint8_t>(y)};
  BOOST_TEST(ybits8.to_string() == "11101110");
  BOOST_TEST(ybits8.to_string() == not_x_plus_1bits8.to_string());
}

// cf. https://youtu.be/NLKQEOgBAnw?t=399
// Logical right shift fills in zero for sign bit.
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(LogicalRightShiftOnNegativeNumbers)
{
  SuperBitSet<8> xbits8 {bit_cast<uint8_t>(logical_right_shift<int8_t>(23, 1))};
  BOOST_TEST(xbits8.to_string() == "00001011");

  int8_t y {-23};
  SuperBitSet<8> ybits8 {bit_cast<uint8_t>(y)};
  BOOST_TEST(ybits8.to_string() == "11101001");

  SuperBitSet<8> zbits8 {bit_cast<uint8_t>(logical_right_shift<int8_t>(y, 1))};
  BOOST_TEST(zbits8.to_string() == "01110100");
}

// cf. https://youtu.be/NLKQEOgBAnw?t=399
// Arithmetic right shift shifts everything to the right including sign bit,
// fills in sign bit with original sign bit.
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ArithmeticRightShiftOnNegativeNumbers)
{
  int8_t x {-23};
  int8_t y {1};
  // Works, but it warns. Narrowing conversion.
  //int8_t z {x >> y};
  //SuperBitSet<8> xbits8 {bit_cast<uint8_t>(z)};
  //BOOST_TEST(xbits8.to_string() == "11110100");
  {
    int8_t x {-22};
    SuperBitSet<8> xbits8 {bit_cast<uint8_t>(x)};
    BOOST_TEST(xbits8.to_string() == "11101010");
    // Works but it warns. Narrowing conversion.
    //int8_t y {x >> 1};
    //SuperBitSet<8> ybits8 {bit_cast<uint8_t>(y)};
    //BOOST_TEST(ybits8.to_string() == "11110101");    
  }
}

// cf. https://en.cppreference.com/w/cpp/language/operator_arithmetic
// lhs << rhs
// left shift of lhs by rhs bits.
// lhs >> rhs
// right shift of lhs by rhs bits
// For built-in operators, lhs and rhs must both have integral or unscoped
// enumeration type. Integral promotions performed on both operands.
// Return type is type of left operand after integral promotions.
// For unsigned a, value of a << b is value a * 2^b, reduced modulo 2^N, where
// N is number of bits in return type (that is, bitwise left shift is performed
// and bits that get shifted out of destination type are discarded).
// For negative a, behavior of a << b is undefined.
//
// since C++14
// For signed and non-negative a, if a*2^b representable in unsigned version
// of return type, then taht value, converted to signed, is value of a << b
// For negative a, value of a >> b is implementation-defined (in most
// implementations, this performs arithmetic right shift, so result remains
// negative)
//
// since C++20
// a << b is unique value congrent to a * 2^b modulo 2^N where N is number of
// bits in return type (that is bitwise left shift performed and bits that get
// shifted out of destination type are discarded).
// Value of a >> b is a/2^b, rounded down (i.e., right shift on signed a is
// arithmetic right shift).

BOOST_AUTO_TEST_SUITE(BitwiseShiftOperator_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(Examples)
{
  char c {0x10};
  SuperBitSet<8> cbits8 {bit_cast<uint8_t>(c)};
  BOOST_TEST(cbits8.to_string() == "00010000");

  SuperBitSet<32> xbits32 {bit_cast<uint32_t>(c << 10)};
  BOOST_TEST(
    xbits32.to_string() == "00000000000000000100000000000000");
  BOOST_TEST((c << 10) == 0x4000);

  unsigned long long ull {0x123};
  SuperBitSet<number_of_bits_in_a_byte * sizeof(unsigned long long)>
    ullbits_ull {bit_cast<uint64_t>(ull)};
  BOOST_TEST(
    ullbits_ull.to_string() ==
      "0000000000000000000000000000000000000000000000000000000100100011");

  {
    SuperBitSet<number_of_bits_in_a_byte * sizeof(unsigned long long)>
      xbits_ull {bit_cast<uint64_t>(ull << 1)};

    BOOST_TEST(
      xbits_ull.to_string() ==
        "0000000000000000000000000000000000000000000000000000001001000110");

    BOOST_TEST(xbits_ull.to_ullong() == 0x246);
  }
  {
    // overflow in unsigned
    SuperBitSet<number_of_bits_in_a_byte * sizeof(unsigned long long)>
      xbits_ull {bit_cast<uint64_t>(ull << 63)};

    BOOST_TEST(
      xbits_ull.to_string() ==
        "1000000000000000000000000000000000000000000000000000000000000000");

    BOOST_TEST(xbits_ull.to_ullong() == 0x8000000000000000);
  }

  // For negative a, value of a >> b in most implementations is arithmetic right
  // shift, shifts everything to right including sign bit, fills in sign bit
  // with original sign bit.
  long long ll {-1000};
  SuperBitSet<number_of_bits_in_a_byte * sizeof(long long)>
    llbits_ll {bit_cast<uint64_t>(ll)};
  BOOST_TEST(
    llbits_ll.to_string() ==
      "1111111111111111111111111111111111111111111111111111110000011000");

  SuperBitSet<number_of_bits_in_a_byte * sizeof(long long)>
    xbits_ll {bit_cast<uint64_t>(ll >> 1)};
  BOOST_TEST(
    xbits_ll.to_string() ==
      "1111111111111111111111111111111111111111111111111111111000001100");
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BitwiseLeftShiftOfNegativeNumbers)
{
  int8_t x {-5};
  SuperBitSet<8> xbits8 {bit_cast<uint8_t>(x)};
  // Indeed this is the "two's complement" of 00000101 = 5; take the complement
  // and then add 1.
  BOOST_TEST(xbits8.to_string() == "11111011");

  // cf. https://en.cppreference.com/w/cpp/language/operator_arithmetic
  // cppreference says this is undefined.
  x << 1;

  auto y {x << 1};

  // std::cout << y << "\n"; // -10

  // template argument deduction/substitution failed.
  // SuperBitSet<8> ybits8 {bit_cast<uint8_t>(x << 1)};
  // SuperBitSet<8> ybits8 {bit_cast<int8_t>(x << 1)};
  //SuperBitSet<16> ybits16 {bit_cast<uint16_t>(x << 1)};
  //SuperBitSet<16> ybits16 {bit_cast<int16_t>(x << 1)};

  SuperBitSet<32> ybits32 {bit_cast<uint32_t>(y)};

  BOOST_TEST(ybits32.to_string() == "11111111111111111111111111110110");
}

BOOST_AUTO_TEST_SUITE_END() // BitwiseShiftOperator_tests

BOOST_AUTO_TEST_SUITE(Masks_tests)

// https://youtu.be/NLKQEOgBAnw?t=444
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BitwiseAndToGetIthBit)
{
  int8_t x {44};

  SuperBitSet<8> xbits8 {bit_cast<uint8_t>(x)};
  BOOST_TEST(xbits8.to_string() == "00101100");

  // (x & (1 << c)) != 0
  BOOST_TEST((x & (1 << 5)) != 0);

  BOOST_TEST((x & (1 << 7)) == 0);
  BOOST_TEST((x & (1 << 6)) == 0);
  BOOST_TEST((x & (1 << 4)) == 0);
  BOOST_TEST((x & (1 << 1)) == 0);
  BOOST_TEST((x & (1 << 0)) == 0);

  BOOST_TEST((x & (1 << 3)) != 0);
  BOOST_TEST((x & (1 << 2)) != 0);

  BOOST_TEST(!is_bit_set_high(x, 0));
  BOOST_TEST(!is_bit_set_high(x, 1));
  BOOST_TEST(!is_bit_set_high(x, 4));
  BOOST_TEST(!is_bit_set_high(x, 6));
  BOOST_TEST(!is_bit_set_high(x, 7));

  BOOST_TEST(is_bit_set_high(x, 2));
  BOOST_TEST(is_bit_set_high(x, 3));
  BOOST_TEST(is_bit_set_high(x, 5));

  int8_t y {-44}; // 11010100

  BOOST_TEST(is_bit_set_high(y, 2));
  BOOST_TEST(is_bit_set_high(y, 4));
  BOOST_TEST(is_bit_set_high(y, 6));
  BOOST_TEST(is_bit_set_high(y, 7));

  BOOST_TEST(!is_bit_set_high(y, 0));
  BOOST_TEST(!is_bit_set_high(y, 1));
  BOOST_TEST(!is_bit_set_high(y, 3));
  BOOST_TEST(!is_bit_set_high(y, 5));
}

// https://youtu.be/NLKQEOgBAnw?t=466
// Set ith bit, x | (1 << i)
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BitwiseOrToSetIthBit)
{
  int8_t x {44};

  BOOST_TEST(is_bit_set_high(x, 5));
  x = set_bit_high(x, 5);
  BOOST_TEST(is_bit_set_high(x, 5));

  BOOST_TEST(!is_bit_set_high(x, 6));
  x = set_bit_high(x, 6);
  BOOST_TEST(is_bit_set_high(x, 6));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ClearBitWithMaskWithAll1ButThatSpot)
{
  int8_t x {44};
  BOOST_TEST(is_bit_set_high(x, 5));
  x = clear_bit(x, 5);
  BOOST_TEST(!is_bit_set_high(x, 5));
}

BOOST_AUTO_TEST_SUITE_END() // Masks_tests

BOOST_AUTO_TEST_SUITE_END() // Radix_tests
BOOST_AUTO_TEST_SUITE_END() // Bits
BOOST_AUTO_TEST_SUITE_END() // Algorithms