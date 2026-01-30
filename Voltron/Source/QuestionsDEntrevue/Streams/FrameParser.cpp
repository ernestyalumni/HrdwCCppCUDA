#include "FrameParser.h"

#include <optional>
#include <span>
#include <vector>

using std::nullopt;
using std::optional;
using std::span;
using std::vector;

namespace QuestionsDEntrevue
{
namespace Streams
{

void FrameParser::push(span<const uint8_t> data)
{
  buffer_.insert(buffer_.end(), data.begin(), data.end());

  if (buffer_.size() > MAX_BUFFER_SIZE)
  {
    buffer_.erase(
      buffer_.begin(),
      buffer_.begin() + buffer_.size() - MAX_BUFFER_SIZE / 2);
  }
}

void FrameParser::push(span<uint8_t> data)
{
  push(span<const uint8_t>(data));
}

optional<vector<uint8_t>> FrameParser::pop_frame(
  const bool is_big_endian)
{
  while (!buffer_.empty())
  {
    // Need at least sync (2) + length (2)
    if (buffer_.size() < PAYLOAD_OFFSET)
    {
      return nullopt;
    }

    // Not starting with sync? Discard front and keep looking. Discard only 1
    // byte on error because second sync byte might be start of a real frame.
    if (buffer_[0] != SYNC_BYTE1 || buffer_[1] != SYNC_BYTE2)
    {
      buffer_.pop_front();
      continue;
    }

    const uint16_t L {bytes_to_uint16(buffer_[2], buffer_[3], is_big_endian)};
  
    // Bogus huge length? Treat as invalid-> discard sync byte
    if (L > MAX_FRAME_SIZE)
    {
      buffer_.pop_front();
      continue;
    }

    // Total frame size if valid
    const size_t frame_length {PAYLOAD_OFFSET + static_cast<size_t>(L) + 2};

    // Not enough bytes yet? Wait for more data.
    if (buffer_.size() < frame_length)
    {
      return nullopt;
    }

    const uint16_t expected_checksum {calculate_checksum(L)};

    const uint16_t received_checksum {
      bytes_to_uint16(
        buffer_[PAYLOAD_OFFSET + L],
        buffer_[PAYLOAD_OFFSET + L + 1],
        is_big_endian)};

    if (expected_checksum != received_checksum)
    {
      // Checksum fail -> discard only first sync byte (maybe next byte is start
      // of a real frame))
      buffer_.pop_front();
      continue;
    }

    // Valid frame! Extract payload
    std::vector<uint8_t> payload {};
    for (size_t i {0}; i < L; ++i)
    {
      payload.push_back(buffer_[PAYLOAD_OFFSET + i]);
    }

    // Consume the entire frame
    buffer_.erase(buffer_.begin(), buffer_.begin() + frame_length);

    return payload;
  }

  return nullopt;
}

uint16_t FrameParser::bytes_to_uint16(
  const uint8_t byte1,
  const uint8_t byte2,
  const bool is_big_endian)
{
  // Big-endian (network byte order, most protocols)
  // buffer_[2] = MSB (most significant byte)
  // buffer_[3] = LSB (least significant byte) 
  if (is_big_endian)
  {
    return static_cast<uint16_t>(byte1) << 8 | static_cast<uint16_t>(byte2);
  }
  // Little-endian (x86/x64 native)
  // buffer_[2] = LSB
  // buffer_[3] = MSB
  else
  {
    return static_cast<uint16_t>(byte1) | static_cast<uint16_t>(byte2) << 8;
  }
}

uint16_t FrameParser::calculate_checksum(
  const uint16_t L,
  const size_t offset)
{
  uint16_t checksum {static_cast<uint16_t>(L)};
  for (size_t i {0}; i < L; ++i)
  {
    checksum ^= buffer_[offset + i];
  }
  return checksum;
}

} // namespace Streams
} // namespace QuestionsDEntrevue