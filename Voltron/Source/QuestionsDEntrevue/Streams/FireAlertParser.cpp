#include "FireAlertParser.h"

#include <cstring> // std::memcpy

namespace QuestionsDEntrevue
{
namespace Streams
{

FireAlertParser::FireAlertParser():
  state_(FIND_TAG),
  tag_buffer_{},
  value_buffer_{},
  last_temperatures_{},
  temperature_index_{0},
  last_pressure_{}
{
  last_temperatures_[0] = 0.0f;
  last_temperatures_[1] = 0.0f;
}

bool FireAlertParser::process_byte(uint8_t byte)
{
  switch (state_)
  {
    case FIND_TAG:
      if (byte == 'P' || byte == 'T')
      {
        tag_buffer_.clear();
        tag_buffer_.push_back(byte);
        state_ = READ_TAG;
      }
      // Ignore any other kinds of bytes while in state FIND_TAG.
      break;
    case READ_TAG:
      tag_buffer_.push_back(byte);
      if ((tag_buffer_.size() == 3) &&
        ((tag_buffer_[0] == 'P') &&
          (tag_buffer_[1] == 'R') &&
          (tag_buffer_[2] == 'S')))
      {
        value_buffer_.clear();
        state_ = READ_PRESSURE;
      }
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