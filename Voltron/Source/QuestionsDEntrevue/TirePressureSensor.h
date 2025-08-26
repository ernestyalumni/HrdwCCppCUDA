#ifndef QUESTIONS_D_ENTREVUE_TIRE_PRESSURE_SENSOR_H
#define QUESTIONS_D_ENTREVUE_TIRE_PRESSURE_SENSOR_H

#include <cstdint>
#include <vector>

namespace QuestionsDEntrevue
{
namespace TirePressureSensor
{

struct I2CLog
{
  uint8_t address {0};
  uint8_t reg {0};
  uint8_t value {0};
  // True if last write succeeded
  bool valid {false};

  static I2CLog& get_instance()
  {
    static I2CLog instance;
    return instance;
  }

  private:

    // Private constructor for singleton.
    I2CLog() = default;
};

//------------------------------------------------------------------------------
/// \param reg The register to write to.
//------------------------------------------------------------------------------
bool i2c_write(uint8_t address, uint8_t reg, uint8_t value);

uint8_t i2c_read(uint8_t address, uint8_t reg);

class PressureSensor
{
  public:
    static constexpr uint8_t I2C_ADDRESS {0x76};
    PressureSensor();
    virtual ~PressureSensor() = default;
  
    int32_t read_pressure();
  
  protected:
    void configure();

  private:
    bool initialized_;
    int32_t calibrationDigP1_;
};

} // namespace TirePressureSensor
} // namespace QuestionsDEntrevue

#endif // QUESTIONS_D_ENTREVUE_TIRE_PRESSURE_SENSOR_H