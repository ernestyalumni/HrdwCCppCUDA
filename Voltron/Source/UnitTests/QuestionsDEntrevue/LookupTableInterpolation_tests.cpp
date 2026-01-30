#include "QuestionsDEntrevue/LookupTableInterpolation.h"

#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <vector>

using QuestionsDEntrevue::interpolate_action;
using std::vector;

BOOST_AUTO_TEST_SUITE(QuestionsDEntrevue)
BOOST_AUTO_TEST_SUITE(LookupTableInterpolation_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateLookupTableInterpolation)
{
  vector<double> velocities {0.0, 10.0, 20.0, 50.0, 100.0};
  vector<double> actions {0.0,  15.0,  35.0,  70.0,  100.0};
  {
    const double target_velocity {-5.0};
    const double result {
      interpolate_action(velocities, actions, target_velocity)};
    // Clamped to first action
    BOOST_TEST(result == 0.0);
  }
  {
    const double target_velocity {0.0};
    const double result {
      interpolate_action(velocities, actions, target_velocity)};
    BOOST_TEST(result == 0.0);
  }
  {
    const double target_velocity {5.0};
    const double result {
      interpolate_action(velocities, actions, target_velocity)};
    BOOST_TEST(result == 7.5);
  }
  {
    const double target_velocity {15.0};
    const double result {
      interpolate_action(velocities, actions, target_velocity)};
    BOOST_TEST(result == 25.0);
  }
  {
    const double target_velocity {100.0};
    const double result {
      interpolate_action(velocities, actions, target_velocity)};
    BOOST_TEST(result == 100.0);
  }
  {
    const double target_velocity {120.0};
    const double result {
      interpolate_action(velocities, actions, target_velocity)};
    // Clamped to last action
    BOOST_TEST(result == 100.0);
  }
}

BOOST_AUTO_TEST_SUITE_END() // LookupTableInterpolation_tests
BOOST_AUTO_TEST_SUITE_END() // QuestionsDEntrevue
