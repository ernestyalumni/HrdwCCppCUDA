#include "Cpp/Classes/BitField.h"
#include "Cpp/Utilities/SuperBitSet.h"
#include "Utilities/ToHexString.h"

#include <boost/test/unit_test.hpp>
#include <sstream> // std::stringstream

using Cpp::Classes::Char4Bits;
using Cpp::Classes::Int4Bits;
using Cpp::Classes::UnsignedChar4Bits;
using Cpp::Classes::UnsignedInt24Bits;
using Cpp::Classes::UnsignedInt4Bits;
using Cpp::Classes::UnsignedInt5Bits;
using Cpp::Utilities::SuperBitSet;
using Utilities::to_hex_string;
using std::stringstream;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(Bits)

BOOST_AUTO_TEST_SUITE(BitField_tests)

BOOST_AUTO_TEST_SUITE(Char4Bits_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CyclesAccordingTo4Bits)
{
  stringstream out;

  Char4Bits a {0};

  BOOST_TEST((a.bits == 0));

  out << to_hex_string(a.bits);

  BOOST_TEST(out.str() == "00");

  a.bits = 1;
  BOOST_TEST((a.bits == 1));

  // Clear the contents of a stringstream.
  // https://stackoverflow.com/questions/20731/how-do-you-clear-a-stringstream-variable
  out.str("");

  out << to_hex_string(a.bits);

  BOOST_TEST(out.str() == "01");

  a.bits = 5;
  BOOST_TEST((a.bits == 5));
  out.str("");
  out << to_hex_string(a.bits);
  BOOST_TEST(out.str() == "05");

  a.bits = 7;
  BOOST_TEST((a.bits == 7));
  out.str("");
  out << to_hex_string(a.bits);
  BOOST_TEST(out.str() == "07");

  a.bits = 8;

  BOOST_TEST((a.bits == -8));
  out.str("");
  out << to_hex_string(a.bits);
  BOOST_TEST(out.str() == "f8");

  a.bits = 9;
  BOOST_TEST((a.bits == -7));
  out.str("");
  out << to_hex_string(a.bits);
  BOOST_TEST(out.str() == "f9");

  a.bits = 14;
  BOOST_TEST((a.bits == -2));
  out.str("");
  out << to_hex_string(a.bits);
  BOOST_TEST(out.str() == "fe");

  a.bits = 15;
  BOOST_TEST((a.bits == -1));
  out.str("");
  out << to_hex_string(a.bits);
  BOOST_TEST(out.str() == "ff");

  // Overflow warning.
  /*
  a.bits = 16;
  BOOST_TEST((a.bits == 0));
  out.str("");
  out << to_hex_string(a.bits);
  BOOST_TEST(out.str() == "00");
  */
}

BOOST_AUTO_TEST_SUITE_END() // Char4Bits_tests


BOOST_AUTO_TEST_SUITE(UnsignedInt4Bits_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CyclesAccordingTo4Bits)
{
  stringstream out;

  UnsignedInt4Bits a {0};
  BOOST_TEST((a.bits == 0));
  out.str("");
  out << to_hex_string(a.bits);
  BOOST_TEST(out.str() == "00000000");
  BOOST_TEST(SuperBitSet<4>{a.bits}.to_string() == "0000");

  a.bits = 1;
  BOOST_TEST((a.bits == 1));
  out.str("");
  out << to_hex_string(a.bits);
  BOOST_TEST(out.str() == "00000001");
  BOOST_TEST(SuperBitSet<4>{a.bits}.to_string() == "0001");

  a.bits = 2;
  BOOST_TEST((a.bits == 2));
  out.str("");
  out << to_hex_string(a.bits);
  BOOST_TEST(out.str() == "00000002");
  BOOST_TEST(SuperBitSet<4>{a.bits}.to_string() == "0010");

  a.bits = 9;
  BOOST_TEST((a.bits == 9));
  out.str("");
  out << to_hex_string(a.bits);
  BOOST_TEST(out.str() == "00000009");
  BOOST_TEST(SuperBitSet<4>{a.bits}.to_string() == "1001");

  a.bits = 10;
  BOOST_TEST((a.bits == 10));
  out.str("");
  out << to_hex_string(a.bits);
  BOOST_TEST(out.str() == "0000000a");
  BOOST_TEST(SuperBitSet<4>{a.bits}.to_string() == "1010");

  a.bits = 11;
  BOOST_TEST((a.bits == 11));
  out.str("");
  out << to_hex_string(a.bits);
  BOOST_TEST(out.str() == "0000000b");
  BOOST_TEST(SuperBitSet<4>{a.bits}.to_string() == "1011");

  a.bits = 14;
  BOOST_TEST((a.bits == 14));
  out.str("");
  out << to_hex_string(a.bits);
  BOOST_TEST(out.str() == "0000000e");
  BOOST_TEST(SuperBitSet<4>{a.bits}.to_string() == "1110");

  a.bits = 15;
  BOOST_TEST((a.bits == 15));
  out.str("");
  out << to_hex_string(a.bits);
  BOOST_TEST(out.str() == "0000000f");
  BOOST_TEST(SuperBitSet<4>{a.bits}.to_string() == "1111");

  // Overflow warning.
  // a.bits = 16;
}

BOOST_AUTO_TEST_SUITE_END() // UnsignedInt4Bits_tests

BOOST_AUTO_TEST_SUITE(UnsignedInt24Bits_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CyclesAccordingTo4Bits)
{

}

BOOST_AUTO_TEST_SUITE_END() // UnsignedInt24Bits_tests


BOOST_AUTO_TEST_SUITE_END() // BitField_tests

BOOST_AUTO_TEST_SUITE_END() // Bits
BOOST_AUTO_TEST_SUITE_END() // Algorithms