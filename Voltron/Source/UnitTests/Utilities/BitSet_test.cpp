//------------------------------------------------------------------------------
// \file BitSet_test.cpp
//------------------------------------------------------------------------------
#include "Utilities/BitCast.h"

#include <bitset>
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <cmath>

using Utilities::number_of_bits_in_a_byte;

BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(BitSet_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(NumberOfBitsInAByteIs8)
{
  BOOST_TEST(number_of_bits_in_a_byte == 8);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PointersToFunctionsStaticCastReinterpretCasts)
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

BOOST_AUTO_TEST_SUITE_END() // BitSet_tests
BOOST_AUTO_TEST_SUITE_END() // Utilities