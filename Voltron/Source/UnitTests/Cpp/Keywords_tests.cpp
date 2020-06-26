//------------------------------------------------------------------------------
// \file Keywords_tests.cpp
//------------------------------------------------------------------------------
#include "Cpp/Utilities/SuperBitSet.h"
#include "Cpp/Utilities/TypeSupport/UnderlyingTypes.h"

#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <iostream>

using Cpp::Utilities::SuperBitSet;
using Cpp::Utilities::TypeSupport::get_underlying_value;
using Cpp::Utilities::number_of_bits_in_a_byte;

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Keywords_tests)

// cf. https://www.tutorialspoint.com/where-are-static-variables-stored-in-c-cplusplus

int func()
{
  static int i {4};
  i++;
  return i;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// cf. https://www.tutorialspoint.com/where-are-static-variables-stored-in-c-cplusplus
BOOST_AUTO_TEST_CASE(DemonstrateStaticVariables)
{
  // In func(), is a static variable, stored in initialized data segment.
  // In main() function, func() returns static variable; it remains in memory
  // because it's static while program is running and provides consistent
  // values.
  {
    BOOST_TEST(func() == 5);
    BOOST_TEST(func() == 6);
    BOOST_TEST(func() == 7);
    BOOST_TEST(func() == 8);
    BOOST_TEST(func() == 9);
    BOOST_TEST(func() == 10);
  }
  BOOST_TEST(func() == 11);
  BOOST_TEST(func() == 12);
  BOOST_TEST(func() == 13);
}

BOOST_AUTO_TEST_SUITE(Casting_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CastingBetweenCharsAndUint8_tAreInverses)
{
  std::cout << "\n\n Casting Tests begin\n";

  {
    const char c {'a'};

    const uint8_t cu {static_cast<uint8_t>(c)};

    std::cout << "cu: " << cu << "\n";


    BOOST_TEST(c == static_cast<char>(cu));
  }


  std::cout << "\n Casting Tests end\n";
}

enum class BitToCharEnum : char
{
  a = 0b00,
  b = 0b01,
  c = 0b10,
  d = 0b11,
  e = 0b101
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CastingEnumToCharHasInverses)
{
  {
    const uint8_t u {0b00};
    const char cu {static_cast<char>(u)};

    BOOST_TEST((static_cast<BitToCharEnum>(cu) == BitToCharEnum::a));
  }
  {
    const uint8_t u {0b01};
    const char cu {static_cast<char>(u)};

    BOOST_TEST((static_cast<BitToCharEnum>(cu) == BitToCharEnum::b));
  }
  {
    const uint8_t u {0b10};
    const char cu {static_cast<char>(u)};

    BOOST_TEST((static_cast<BitToCharEnum>(cu) == BitToCharEnum::c));
  }
  {
    const uint8_t u {0b11};
    const char cu {static_cast<char>(u)};

    BOOST_TEST((static_cast<BitToCharEnum>(cu) == BitToCharEnum::d));
  }
  {
    const uint8_t u {0b100};
    BOOST_TEST_REQUIRE(u > 0b11);
    const char cu {static_cast<char>(u)};

    BOOST_TEST((static_cast<BitToCharEnum>(cu) != BitToCharEnum::a));
    BOOST_TEST((static_cast<BitToCharEnum>(cu) != BitToCharEnum::b));
    BOOST_TEST((static_cast<BitToCharEnum>(cu) != BitToCharEnum::c));
    BOOST_TEST((static_cast<BitToCharEnum>(cu) != BitToCharEnum::d));
  }
  {
    const uint8_t u {0b101};
    BOOST_TEST_REQUIRE(u > 0b11);
    const char cu {static_cast<char>(u)};

    BOOST_TEST((static_cast<BitToCharEnum>(cu) != BitToCharEnum::a));
    BOOST_TEST((static_cast<BitToCharEnum>(cu) != BitToCharEnum::b));
    BOOST_TEST((static_cast<BitToCharEnum>(cu) != BitToCharEnum::c));
    BOOST_TEST((static_cast<BitToCharEnum>(cu) != BitToCharEnum::d));

    BOOST_TEST(get_underlying_value(BitToCharEnum::a) < cu);
    BOOST_TEST(get_underlying_value(BitToCharEnum::b) < cu);
    BOOST_TEST(get_underlying_value(BitToCharEnum::c) < cu);
    BOOST_TEST(get_underlying_value(BitToCharEnum::d) < cu);
  }
  {
    const uint8_t u {0b111};
    const char cu {static_cast<char>(u)};

    const BitToCharEnum outside {static_cast<BitToCharEnum>(cu)};

    BOOST_TEST(
      get_underlying_value(outside) > get_underlying_value(BitToCharEnum::e));
  }
}

BOOST_AUTO_TEST_SUITE_END() // Casting_tests

BOOST_AUTO_TEST_SUITE_END() // Keywords_tests
BOOST_AUTO_TEST_SUITE_END() // Cpp