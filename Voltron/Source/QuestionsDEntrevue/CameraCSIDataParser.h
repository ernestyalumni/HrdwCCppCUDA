#ifndef QUESTIONS_D_ENTREVUE_CAMERA_CSI_DATA_PARSER_H
#define QUESTIONS_D_ENTREVUE_CAMERA_CSI_DATA_PARSER_H

#include <cstdint>
#include <vector>

namespace QuestionsDEntrevue
{

//------------------------------------------------------------------------------
/// \brief Parse the camera CSI data.
/// \details Imagine you're given a CSI (Camera Serial Interface) input line
/// with raw 8-bit packet data representing grayscale values from a rolling
/// shutter camera. Each frame starts with a 2-byte magic number: 0xFF 0xD8 and
/// ends with 0xFF 0xD9. Between these, each byte is a pixel value.
///
/// Write a function that, given a stream of bytes, extracts all full frames and
/// returns them as a vector of vectors of uint8_t.
//------------------------------------------------------------------------------
std::vector<std::vector<uint8_t>> parse_camera_csi_data(
  const uint8_t* data,
  size_t length);

namespace CSICamera
{

struct Frame
{
  uint16_t frame_id_ {};
  std::vector<uint8_t> payload_{};
};

enum FrameType
{
  DT_FS = 0x00,
  DT_FE = 0x01,
  DT_RAW8 = 0x2A
};

std::vector<Frame> parse_csi2(const uint8_t* buffer, size_t n);

} // namespace CSICamera

} // namespace QuestionsDEntrevue

#endif // QUESTIONS_D_ENTREVUE_CAMERA_CSI_DATA_PARSER_H
