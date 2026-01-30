#ifndef QUESTIONS_D_ENTREVUE_STREAMS_FRAME_PARSER_H
#define QUESTIONS_D_ENTREVUE_STREAMS_FRAME_PARSER_H

#include <cstdint>
#include <deque>
#include <optional>
#include <span>
#include <vector>

namespace QuestionsDEntrevue
{
namespace Streams
{

//------------------------------------------------------------------------------
/// You receive bytes from a device over UART/RS-485 in arbitrary chunk sizes. Frames look like:
///
/// Sync word: 0xAA 0x55
/// Length: 2 bytes L (payload length)
/// Payload: L bytes
/// Checksum: 2 bytes = XOR of L and all payload bytes
//------------------------------------------------------------------------------
class FrameParser
{
  public:

    static constexpr size_t MAX_BUFFER_SIZE {4096};
    static constexpr size_t MAX_FRAME_SIZE {1440};

    static constexpr uint8_t SYNC_BYTE1 {0xAA};
    static constexpr uint8_t SYNC_BYTE2 {0x55};

    static constexpr size_t PAYLOAD_OFFSET {sizeof(uint16_t) * 2};

    //--------------------------------------------------------------------------
    /// https://en.cppreference.com/w/cpp/container/span.html
    /// span - "an object that can refer to a continguous sequence of objects
    /// with the first element of the sequence at position 0".
    /// Accept any contiguous range of bytes.
    //--------------------------------------------------------------------------
    void push(std::span<const uint8_t> data);
    void push(std::span<uint8_t> data);

    std::optional<std::vector<uint8_t>> pop_frame(
      const bool is_big_endian = false);
    
    inline size_t get_buffered_bytes() const
    {
      return buffer_.size();
    }

    uint16_t bytes_to_uint16(
      const uint8_t byte1,
      const uint8_t byte2,
      const bool is_big_endian = false);

    // Compute expected checksum: XOR(L + payload[0..L-1])
    uint16_t calculate_checksum(
      const uint16_t L,
      const size_t offset = PAYLOAD_OFFSET);

  private:

    std::deque<uint8_t> buffer_;
};
  
} // namespace Streams
} // namespace QuestionsDEntrevue

#endif // QUESTIONS_D_ENTREVUE_STREAMS_FRAME_PARSER_H