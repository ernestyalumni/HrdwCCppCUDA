//------------------------------------------------------------------------------
/// \file Radix_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://en.wikipedia.org/wiki/Radix
/// \details To run only the Bits unit tests, do this:
/// ./Check --run_test="Algorithms/Bits"
//------------------------------------------------------------------------------
#include "Cpp/Utilities/SuperBitSet.h"

#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <iostream>
#include <limits>

using Cpp::Utilities::SuperBitSet;
using Cpp::Utilities::number_of_bits_in_a_byte;

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
    std::cout << "std::numeric_limits<uint16_t>::max(): " <<
      std::numeric_limits<uint16_t>::max() << "\n";
    std::cout << "std::numeric_limits<uint32_t>::max(): " <<
      std::numeric_limits<uint32_t>::max() << "\n";
    std::cout << "std::numeric_limits<uint64_t>::max(): " <<
      std::numeric_limits<uint64_t>::max() << "\n";

    std::cout << "\n FloatingPoint limits \n";

    std::cout << "std::numeric_limits<float>::max(): " <<
      std::numeric_limits<float>::max() << "\n";
    std::cout << "std::numeric_limits<double>::max(): " <<
      std::numeric_limits<double>::max() << "\n";
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


BOOST_AUTO_TEST_SUITE_END() // Radix_tests
BOOST_AUTO_TEST_SUITE_END() // Bits
BOOST_AUTO_TEST_SUITE_END() // Algorithms