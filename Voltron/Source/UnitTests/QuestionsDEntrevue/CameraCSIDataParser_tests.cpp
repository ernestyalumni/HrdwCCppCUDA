#include "QuestionsDEntrevue/CameraCSIDataParser.h"

#include <boost/test/unit_test.hpp>
#include <cstdint>

using QuestionsDEntrevue::parse_camera_csi_data;
using QuestionsDEntrevue::CSICamera::parse_csi2;

BOOST_AUTO_TEST_SUITE(Entrevue)
BOOST_AUTO_TEST_SUITE(CameraCSIDataParser_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ParseCameraCSIDataWorks)
{
  const uint8_t buf[]
  {
    0x00, 0x11,                         // noise
    0xFF, 0xD8, 0x10, 0x20, 0x30,       // Frame 1 start + 3 pixels
    0xFF, 0xD9,                         // Frame 1 end
    0xAA,                               // noise
    0xFF, 0xD8, 0x01, 0x02, 0x03, 0x04, // Frame 2 start + 4 pixels
    0xFF, 0xD9,
    0xFF, 0xD8, 0x99, 0x88              // partial frame, should be ignored
  };

  const auto frames = parse_camera_csi_data(buf, sizeof(buf));
  BOOST_TEST(frames.size() == 2);
  BOOST_TEST(frames[0].size() == 5);
  BOOST_TEST(frames[1].size() == 6);
  BOOST_TEST(frames[0][0] == 0xFF);
  BOOST_TEST(frames[0][1] == 0xD8);
  BOOST_TEST(frames[0][2] == 0x10);
  BOOST_TEST(frames[0][3] == 0x20);
  BOOST_TEST(frames[0][4] == 0x30);
  BOOST_TEST(frames[1][0] == 0xFF);
  BOOST_TEST(frames[1][1] == 0xD8);
  BOOST_TEST(frames[1][2] == 0x01);
  BOOST_TEST(frames[1][3] == 0x02);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ParseCSI2Works)
{
    // Frame 0x0001: FS, RAW8(4 bytes), FE
    // Header: [DT, WC_MSB, WC_LSB, ECC(ignored)]
    const uint8_t fs1[] {0x00, 0x00, 0x01, 0x00}; // FS, frame_id=0x0001, CRC16=0x0000
    const uint8_t lp1[] {0x2A, 0x00, 0x04, 0x00, 0x10, 0x20, 0x30, 0x40, 0x00, 0x00}; // RAW8 len=4 + CRC16 dummy
    const uint8_t fe1[] {0x01, 0x00, 0x01, 0x00}; // FE, frame_id=0x0001

    // Frame 0x0002: FS, RAW8(3 bytes), RAW8(2 bytes), FE
    const uint8_t fs2[] {0x00, 0x00, 0x02, 0x00};
    const uint8_t lp2a[] {0x2A, 0x00, 0x03, 0x00, 0xAA, 0xBB, 0xCC, 0x00, 0x00};
    const uint8_t lp2b[] {0x2A, 0x00, 0x02, 0x00, 0x55, 0x66, 0x00, 0x00};
    const uint8_t fe2[] {0x01, 0x00, 0x02, 0x00};

    std::vector<uint8_t> buf {};
    // noise
    buf.push_back(0xFF); buf.push_back(0xEE);
    // frame 1
    buf.insert(buf.end(), std::begin(fs1), std::end(fs1));
    buf.insert(buf.end(), std::begin(lp1), std::end(lp1));
    buf.insert(buf.end(), std::begin(fe1), std::end(fe1));
    // frame 2
    buf.insert(buf.end(), std::begin(fs2), std::end(fs2));
    buf.insert(buf.end(), std::begin(lp2a), std::end(lp2a));
    buf.insert(buf.end(), std::begin(lp2b), std::end(lp2b));
    buf.insert(buf.end(), std::begin(fe2), std::end(fe2));
    // partial tail (ignored)
    buf.push_back(0x2A); buf.push_back(0x00); buf.push_back(0x10); buf.push_back(0x00); // long pkt header only


}

BOOST_AUTO_TEST_SUITE_END() // CameraCSIDataParser_tests
BOOST_AUTO_TEST_SUITE_END() // Entrevue
