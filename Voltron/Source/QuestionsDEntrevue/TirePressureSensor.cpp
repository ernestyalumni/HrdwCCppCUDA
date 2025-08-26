#include "TirePressureSensor.h"

#include <cstdint>

namespace QuestionsDEntrevue
{
namespace TirePressureSensor
{

bool i2c_write(uint8_t address, uint8_t reg, uint8_t value)
{
  // std::cout << "Mock write: Addr " << static_cast<int>(address) << ", Reg " <<
  //   static_cast<int>(reg) << ", Value " << static_cast<int>(value) << std::endl;

  I2CLog& log {I2CLog::get_instance()};
  log.address = address;
  log.reg = reg;
  log.value = value;
  log.valid = true;

  return true;
}

uint8_t i2c_read(uint8_t address, uint8_t reg)
{
  // Status: Not busy, no error
  if (reg == 0xF3)
  {
    return 0x00;
  }
  // Returning fixed value for testing, pressure=0x800000 for 1013 hPa
  // MSB
  if (reg == 0xF7)
  {
    return 0x80;
  }
  // LSB
  if (reg == 0xF8)
  {
    return 0x00;
  }
  // XLSB
  if (reg == 0xF9)
  {
    return 0x00;
  }
  return 0;
}

PressureSensor::PressureSensor():
  initialized_{false},
  // Fixed value for simulation.
  calibrationDigP1_{36477}
{
  configure();
  initialized_ = true;
}

void PressureSensor::configure()
{
  i2c_write(I2C_ADDRESS, 0xF5, 0x20);
  i2c_write(I2C_ADDRESS, 0xF4, 0x2F);
}

int32_t PressureSensor::read_pressure()
{
  if (!initialized_)
  {
    return -1;
  }

  // Volatile for hardware. volatile tells compiler; don't optimize read
  // writes to this variable-it may change outside the program's control.
  volatile uint8_t status {i2c_read(I2C_ADDRESS, 0xF3)};
  // Bit[0]: Broken sensor error.
  if (status & 0x01)
  {
    configure();
    status = i2c_read(I2C_ADDRESS, 0xF3);
    // Fallback to default sea level pressure.
    if (status & 0x01)
    {
      return 1013;
    }
  }

  // Bit[3]: Busy - wait (mock no wait)
  if (status & 0x08)
  {
    return -1;
  }

  // Read 20-bit pressure (avoid undefined behavior: unsigned, bounds)
  uint32_t raw {
    (static_cast<uint32_t>(i2c_read(I2C_ADDRESS, 0xF7)) << 12) |
    (static_cast<uint32_t>(i2c_read(I2C_ADDRESS, 0xF8)) << 4) |
    (static_cast<uint32_t>(i2c_read(I2C_ADDRESS, 0xF9)) >> 4)
  };

  // Bound check.
  if (raw == 0 || raw > 0xFFFFF)
  {
    return -1;
  }

  // Simple compensation (real: use all cal digs)
  const int32_t pressure {
    static_cast<int32_t>(raw) / 256 + calibrationDigP1_ / 1000
  };

  return pressure;
}

} // namespace TirePressureSensor
} // namespace QuestionsDEntrevue