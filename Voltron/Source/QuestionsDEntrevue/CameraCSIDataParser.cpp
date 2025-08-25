#include "CameraCSIDataParser.h"

#include <cstdint>
#include <vector>

using std::vector;

namespace QuestionsDEntrevue
{

vector<vector<uint8_t>> parse_camera_csi_data(
  const uint8_t* data,
  size_t length)
{
  vector<vector<uint8_t>> frames {};
  vector<uint8_t> current_frame {};
  bool is_in_frame {false};


  for (size_t i {0}; i < length; ++i)
  {
    if (
      !is_in_frame &&
      i + 1 < length &&
      data[i] == 0xFF &&
      data[i + 1] == 0xD8)
    {
      is_in_frame = true;
      current_frame.clear();
      current_frame.push_back(data[i]);
      current_frame.push_back(data[i + 1]);
      i += 1;
    }
    else if (
      is_in_frame &&
      i + 1 < length &&
      data[i] == 0xFF &&
      data[i + 1] == 0xD9)
    {
      is_in_frame = false;
      frames.push_back(current_frame);
      current_frame.clear();
      i += 1;
    }
    else if (is_in_frame)
    {
      current_frame.push_back(data[i]);
    }
  }

  return frames;
}

namespace CSICamera
{

vector<Frame> parse_csi2(const uint8_t* buffer, size_t n)
{
  vector<Frame> frames {};
  Frame current_frame {};
  bool is_in_frame {false};

  size_t i {0};

  while (i + 4 <= n)
  {
    uint8_t dt {buffer[i + 0]};
    uint16_t wc {static_cast<uint16_t>(
      (static_cast<uint16_t>(buffer[i + 1]) << 8) | 
      static_cast<uint16_t>(buffer[i + 2]))};
    i += 4;

    if (dt == DT_FS)
    {
      is_in_frame = true;
      current_frame = Frame{};
      current_frame.frame_id_ = wc;
    }
    else if (dt == DT_FE)
    {
      if (is_in_frame)
      {
        frames.push_back(std::move(current_frame));
        is_in_frame = false;
      }
    }
    // long packet: dt = RAW8
    else
    {
      // Partial packet at end, stop.
      if (i + wc + 2 > n)
      {
        break;
      }
      if (dt == DT_RAW8 && is_in_frame)
      {
        const uint8_t* payload {buffer + i};
        current_frame.payload_.insert(
          current_frame.payload_.end(),
          payload,
          payload + wc);
      }
      i += wc + 2;
    }
  }

  for (size_t i {0}; i < n; ++i)
  {
    if (buffer[i] == 0xFF && buffer[i + 1] == 0xD8)
    {
      // Fix: Create Frame with proper members
      Frame frame{};
      frame.frame_id_ = 0; // or some appropriate frame ID
      frame.payload_.push_back(buffer[i]);
      frame.payload_.push_back(buffer[i + 1]);
      frames.push_back(std::move(frame));
    }
  }

  return frames;
}

} // namespace CSICamera

} // namespace QuestionsDEntrevue

