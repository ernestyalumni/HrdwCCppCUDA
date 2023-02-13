#include "DataStructures/Alphabet.h"

#include <boost/test/unit_test.hpp>
#include <cstddef> // std::size_t
#include <limits>

using DataStructures::Alphabet;
using std::size_t;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Alphabet_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AsciiChartComparesWithUnicode16Literals)
{
  constexpr std::size_t char_maximum {std::numeric_limits<char>::max()};
  BOOST_TEST(char_maximum == 127);

  // See "ASCII Chart" in cppreference.com for "source of truth."
  // https://en.cppreference.com/w/cpp/language/ascii
  BOOST_TEST(static_cast<size_t>('*') == 42);
  BOOST_TEST(static_cast<size_t>('+') == 43);
  BOOST_TEST(static_cast<size_t>(',') == 44);
  BOOST_TEST(static_cast<size_t>('-') == 45);

  BOOST_TEST(static_cast<size_t>('0') == 48);
  BOOST_TEST(static_cast<size_t>('1') == 49);
  BOOST_TEST(static_cast<size_t>('2') == 50);
  BOOST_TEST(static_cast<size_t>('9') == 57);

  BOOST_TEST(static_cast<size_t>('A') == 65);
  BOOST_TEST(static_cast<size_t>('B') == 66);
  BOOST_TEST(static_cast<size_t>('C') == 67);
  BOOST_TEST(static_cast<size_t>('Z') == 90);

  BOOST_TEST(static_cast<size_t>('a') == 97);
  BOOST_TEST(static_cast<size_t>('b') == 98);
  BOOST_TEST(static_cast<size_t>('c') == 99);
  BOOST_TEST(static_cast<size_t>('z') == 122);

  BOOST_TEST(static_cast<size_t>(u'*') == 42);
  BOOST_TEST(static_cast<size_t>(u'+') == 43);
  BOOST_TEST(static_cast<size_t>(u',') == 44);
  BOOST_TEST(static_cast<size_t>(u'-') == 45);

  BOOST_TEST(static_cast<size_t>(u'0') == 48);
  BOOST_TEST(static_cast<size_t>(u'1') == 49);
  BOOST_TEST(static_cast<size_t>(u'2') == 50);
  BOOST_TEST(static_cast<size_t>(u'9') == 57);

  BOOST_TEST(static_cast<size_t>(u'A') == 65);
  BOOST_TEST(static_cast<size_t>(u'B') == 66);
  BOOST_TEST(static_cast<size_t>(u'C') == 67);
  BOOST_TEST(static_cast<size_t>(u'Z') == 90);

  BOOST_TEST(static_cast<size_t>(u'a') == 97);
  BOOST_TEST(static_cast<size_t>(u'b') == 98);
  BOOST_TEST(static_cast<size_t>(u'c') == 99);
  BOOST_TEST(static_cast<size_t>(u'z') == 122);

  const char16_t a_literal {0x0061};
  BOOST_TEST((a_literal == u'a'));
  BOOST_TEST(sizeof(a_literal) == 2);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AlphabetCharacterMaximumIsForChar16)
{
  BOOST_TEST(Alphabet::character_maximum_ == 65535);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AlphabetConstructsFromString)
{
  Alphabet alphabet {"01234567"};
}

BOOST_AUTO_TEST_SUITE_END() // Alphabet_tests
BOOST_AUTO_TEST_SUITE_END() // DataStructures