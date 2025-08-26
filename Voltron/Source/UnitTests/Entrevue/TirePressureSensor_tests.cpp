#include "QuestionsDEntrevue/TirePressureSensor.h"

#include <boost/test/unit_test.hpp>
#include <cstdint>

using QuestionsDEntrevue::TirePressureSensor::PressureSensor;
using QuestionsDEntrevue::TirePressureSensor::I2CLog;

BOOST_AUTO_TEST_SUITE(Entrevue)
BOOST_AUTO_TEST_SUITE(TirePressureSensor_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PressureSensorConstructs)
{
  PressureSensor sensor {};

  I2CLog& log {I2CLog::get_instance()};
  BOOST_CHECK_EQUAL(log.address, 0x76);
  BOOST_CHECK_EQUAL(log.reg, 0xF4);
  BOOST_CHECK_EQUAL(log.value, 0x2f);
  BOOST_CHECK_EQUAL(log.valid, true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ReadPressureReadsPressure)
{
  PressureSensor sensor {};

  I2CLog& log {I2CLog::get_instance()};
  BOOST_CHECK_EQUAL(log.address, 0x76);
  BOOST_CHECK_EQUAL(log.reg, 0xF4);
  BOOST_CHECK_EQUAL(log.value, 0x2f);
  BOOST_CHECK_EQUAL(log.valid, true);

  BOOST_CHECK_EQUAL(sensor.read_pressure(), 2084);
}

BOOST_AUTO_TEST_SUITE_END() // TirePressureSensor_tests
BOOST_AUTO_TEST_SUITE_END() // Entrevue
