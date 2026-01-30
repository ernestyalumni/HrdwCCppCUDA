#include "QuestionsDEntrevue/Streams/FrameParser.h"

#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <span>
#include <vector>

using QuestionsDEntrevue::Streams::FrameParser;
using std::vector;

BOOST_AUTO_TEST_SUITE(QuestionsDEntrevue)
BOOST_AUTO_TEST_SUITE(Streams)
BOOST_AUTO_TEST_SUITE(FrameParser_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FrameParserParsesOutValidFrames)
{
  FrameParser parser {};

  // Simulated noisy stream with two frames + garbage
  uint8_t stream[] = {
    // garbage
    0x12, 0x34,
    // frame 1: payload 11 22 33, chk=0x00 (correct)
    0xAA, 0x55, 0x00, 0x03, 0x11, 0x22, 0x33, 0x00, 0x03,
    // noise
    0xFF, 0xFE,
    // frame 2: payload AB CD, chk should be 0x66 (correct)
    0xAA, 0x55, 0x00, 0x02, 0xAB, 0xCD, 0x00, 0x64,
    // partial frame (incomplete)
    0xAA, 0x55, 0x01,
  };

  BOOST_TEST(parser.bytes_to_uint16(0x00, 0x03, true) == 0x0003);
  BOOST_TEST(parser.bytes_to_uint16(0x00, 0x02, true) == 0x0002);

  parser.push(std::span<uint8_t>(stream));
  BOOST_TEST(parser.get_buffered_bytes() == sizeof(stream));

  BOOST_TEST(parser.calculate_checksum(3, 6) == 0x03);
  BOOST_TEST(parser.calculate_checksum(2, 17) == 0x64);

  const auto frame1 = parser.pop_frame(true);
  BOOST_TEST(parser.get_buffered_bytes() == sizeof(stream) - 2 - 4 - 3 - 2);
  BOOST_TEST(frame1.has_value());
  BOOST_TEST(frame1.value().size() == 3);
  BOOST_TEST(frame1.value()[0] == 0x11);
  BOOST_TEST(frame1.value()[1] == 0x22);
  BOOST_TEST(frame1.value()[2] == 0x33);

  const auto frame2 = parser.pop_frame(true);
  BOOST_TEST(
    parser.get_buffered_bytes() == sizeof(stream) - 11 - 2 - 4 - 2 - 2);
  BOOST_TEST(frame2.has_value());
  BOOST_TEST(frame2.value().size() == 2);
  BOOST_TEST(frame2.value()[0] == 0xAB);
  BOOST_TEST(frame2.value()[1] == 0xCD);

  BOOST_TEST(parser.get_buffered_bytes() == 3);

  const auto frame3 = parser.pop_frame(true);
  BOOST_TEST(!frame3.has_value());

  const auto frame4 = parser.pop_frame(true);
  BOOST_TEST(!frame4.has_value());
}

BOOST_AUTO_TEST_SUITE_END() // FrameParser_tests
BOOST_AUTO_TEST_SUITE_END() // Streams
BOOST_AUTO_TEST_SUITE_END() // QuestionsDEntrevue
