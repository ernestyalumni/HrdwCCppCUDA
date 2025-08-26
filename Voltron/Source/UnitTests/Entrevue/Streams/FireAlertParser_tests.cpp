#include "QuestionsDEntrevue/Streams/FireAlertParser.h"

#include <boost/test/unit_test.hpp>
#include <cstdint>

using QuestionsDEntrevue::Streams::FireAlertParser;

BOOST_AUTO_TEST_SUITE(Entrevue)
BOOST_AUTO_TEST_SUITE(Streams)
BOOST_AUTO_TEST_SUITE(FireAlertParser_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FireAlertParserConstructs)
{
  uint8_t stream[] = {
    'T', 'E', 'M', 'P',
    0x42, 0x78, 0x00, 0x00,
    'P', 'R', 'S',
    0x42, 0x78, 0x00, 0x00,
    'T', 'E', 'M', 'P',
    0x42, 0x8C, 0x00, 0x00,
    'P', 'R', 'S', 0x00
  };

  FireAlertParser parser {};

  bool alert {false};

  for (uint8_t byte : stream)
  {
    alert = parser.process_byte(byte);
    if (alert)
    {
      break;
    }
  }

  BOOST_CHECK_EQUAL(alert, true);
  BOOST_CHECK_EQUAL(parser.get_last_pressure(), 2084);
  BOOST_CHECK_EQUAL(parser.get_state(), FireAlertParser::State::FIND_TAG);
}

BOOST_AUTO_TEST_SUITE_END() // FireAlertParser_tests
BOOST_AUTO_TEST_SUITE_END() // Streams
BOOST_AUTO_TEST_SUITE_END() // Entrevue