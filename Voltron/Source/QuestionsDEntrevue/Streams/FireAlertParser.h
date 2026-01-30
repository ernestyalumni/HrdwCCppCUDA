#ifndef QUESTIONS_D_ENTREVUE_STREAMS_FIRE_ALERT_PARSER_H
#define QUESTIONS_D_ENTREVUE_STREAMS_FIRE_ALERT_PARSER_H

#include <array>
#include <cstdint>
#include <vector>

namespace QuestionsDEntrevue
{
namespace Streams
{

class FireAlertParser
{
  public:

    static constexpr float TEMPERATURE_THRESHOLD {50.0f};

    // Tag constants (as arrays for easy comparison)
    static constexpr std::array<uint8_t, 3> PRESSURE_TAG {{'P', 'R', 'S'}};
    static constexpr std::array<uint8_t, 4> TEMPERATURE_TAG {
      {'T', 'E', 'M', 'P'}};

    enum State
    {
      FIND_TAG,
      READ_TAG,
      READ_PRESSURE,
      READ_TEMPERATURE,
    };

    FireAlertParser();
    virtual ~FireAlertParser() = default;

    bool process_byte(uint8_t byte, const bool is_big_endian=true);

    inline State get_state() const
    {
      return state_;
    }

    inline int32_t get_last_pressure() const
    {
      return last_pressure_;
    }

  protected:

    bool check_to_alert() const
    {
      return (temperature_index_ == 0) &&
        (last_temperatures_[0] > TEMPERATURE_THRESHOLD) &&
        (last_temperatures_[1] > TEMPERATURE_THRESHOLD);
    }

  private:

    void reset_to_find_tag()
    {
      state_ = FIND_TAG;
      tag_buffer_.clear();
      value_buffer_.clear();
    }

    State state_;

    std::vector<uint8_t> tag_buffer_;
    std::vector<uint8_t> value_buffer_;

    float last_temperatures_[2];
    // 0 or 1 for circular buffer
    int temperature_index_;

    int32_t last_pressure_;
};

} // namespace Streams
} // namespace QuestionsDEntrevue

#endif // QUESTIONS_D_ENTREVUE_STREAMS_FIRE_ALERT_PARSER_H