//------------------------------------------------------------------------------
// \file BitSet_test.cpp
//------------------------------------------------------------------------------
#include "Utilities/BitSet.h"

#include <bitset>
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <iostream>
#include <string>
// https://stackoverflow.com/questions/20731/how-do-you-clear-a-stringstream-variable
#include <sstream>

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

BOOST_AUTO_TEST_SUITE_END() // BitSet_tests
BOOST_AUTO_TEST_SUITE_END() // Utilities