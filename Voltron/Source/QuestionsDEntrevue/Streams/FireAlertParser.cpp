#include "FireAlertParser.h"

#include <cstring> // std::memcpy

namespace QuestionsDEntrevue
{
namespace Streams
{

//------------------------------------------------------------------------------
/// Helper: Convert 4 bytes to uint32_t
//------------------------------------------------------------------------------
static uint32_t bytes_to_uint32(
  const std::vector<uint8_t>& bytes,
  size_t offset,
  const bool is_big_endian)
{
  if (is_big_endian)
  {
    return (static_cast<uint32_t>(bytes[offset]) << 24) |
            (static_cast<uint32_t>(bytes[offset + 1]) << 16) |
            (static_cast<uint32_t>(bytes[offset + 2]) << 8) |
            static_cast<uint32_t>(bytes[offset + 3]);
  }
  else
  {
    return (static_cast<uint32_t>(bytes[offset]) |
            (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
            (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
            static_cast<uint32_t>(bytes[offset + 3]) << 24);
  }
}

FireAlertParser::FireAlertParser():
  state_(FIND_TAG),
  tag_buffer_{},
  value_buffer_{},
  last_temperatures_{0.0f, 0.0f},
  temperature_index_{0},
  last_pressure_{}
{}

bool FireAlertParser::process_byte(uint8_t byte, const bool is_big_endian)
{
  switch (state_)
  {
    case FIND_TAG:
      if (byte == PRESSURE_TAG[0] || byte == TEMPERATURE_TAG[0])
      {
        tag_buffer_.clear();
        tag_buffer_.push_back(byte);
        state_ = READ_TAG;
      }
      // Ignore any other kinds of bytes while in state FIND_TAG.
      break;
    case READ_TAG:
      tag_buffer_.push_back(byte);
      if ((tag_buffer_.size() == PRESSURE_TAG.size()) &&
        ((tag_buffer_[0] == 'P') &&
          (tag_buffer_[1] == 'R') &&
          (tag_buffer_[2] == 'S')))
      {
        value_buffer_.clear();
        state_ = READ_PRESSURE;
      }
      // Check if we've matched TEMPERATURE tag ("TEMP")
      else if ((tag_buffer_.size() == 4) &&
        ((tag_buffer_[0] == 'T') &&
          (tag_buffer_[1] == 'E') &&
          (tag_buffer_[2] == 'M') &&
          (tag_buffer_[3] == 'P')))
      {
        value_buffer_.clear();
        state_ = READ_TEMPERATURE;
      }
      else if (tag_buffer_.size() > 4)
      {
        reset_to_find_tag();
      }
      break;
    case READ_PRESSURE:
      value_buffer_.push_back(byte);
      if (value_buffer_.size() == 4)
      {
        uint32_t be_value {bytes_to_uint32(value_buffer_, 0, is_big_endian)};
        last_pressure_ = static_cast<int32_t>(be_value);
      }
      break;
    case READ_TEMPERATURE:
      value_buffer_.push_back(byte);
      if (value_buffer_.size() == 4)
      {
        // Reconstruct a big-endian uint32_t
        uint32_t be_value {
          static_cast<uint32_t>(value_buffer_[0]) << 24 |
          static_cast<uint32_t>(value_buffer_[1]) << 16 |
          static_cast<uint32_t>(value_buffer_[2]) << 8 |
          static_cast<uint32_t>(value_buffer_[3])
        };
        if (state_ == READ_PRESSURE)
        {
          int32_t pressure {};
          std::memcpy(&pressure, &be_value, sizeof(pressure));
        }
        else if (state_ == READ_TEMPERATURE)
        {
          float temperature {};
          std::memcpy(&temperature, &be_value, sizeof(temperature));

          last_temperatures_[temperature_index_] = temperature;
          temperature_index_ = (temperature_index_ + 1) % 2;

          if (temperature_index_ == 0 && check_to_alert())
          {
            reset_to_find_tag();
            return true;
          }
        }
        else
        {
          reset_to_find_tag();
        }
      }
      break;
  }
  return false;
}

} // namespace Streams
} // namespace QuestionsDEntrevue