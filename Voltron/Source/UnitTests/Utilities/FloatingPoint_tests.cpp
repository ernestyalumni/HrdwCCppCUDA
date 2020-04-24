//------------------------------------------------------------------------------
// \file FloatingPoint_tests.cpp
//------------------------------------------------------------------------------
#include "Utilities/FloatingPoint.h"

#include <boost/test/unit_test.hpp>
#include <cmath>
#include <limits>

using Utilities::Conversions::WithUnion::FloatingPointToBitsInUnion;
using Utilities::Conversions::WithUnion::floating_point_to_bitset;

BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(FloatingPoint_tests)

// Significand is also the Mantissa
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DoublePrecisionInSignificand)
{
  BOOST_TEST(std::pow(2, -53) == 1.1102230246251565e-16);
}

BOOST_AUTO_TEST_SUITE(ConversionsWithUnion)

// cf. https://stackoverflow.com/questions/474007/floating-point-to-binary-valuec
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(UnionClassesTakesInFloatingPointValues)
{
  {
    FloatingPointToBitsInUnion::FloatUIntDirectSum data;
    data.input_ = 0.125f;
  }
  {
    FloatingPointToBitsInUnion::DoubleULongDirectSum data;
    data.input_ = 0.03125;
  }

  BOOST_TEST(true);
}

// cf. https://stackoverflow.com/questions/474007/floating-point-to-binary-valuec
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FloatingPointToBitSetTakesInFloats)
{
  {
    const auto result = floating_point_to_bitset<float, sizeof(float)>(
      2.25125f);

    BOOST_TEST(result.to_string() == "01000000000100000001010001111011");
    BOOST_TEST(result[4] == 1);
    BOOST_TEST(result[6] == 1);
    BOOST_TEST(result[7] == 0);
  }
  {
    const auto result = floating_point_to_bitset<float, sizeof(float)>(0.125f);

    BOOST_TEST(result.to_string() == "00111110000000000000000000000000");
  }
  {
    const auto result = floating_point_to_bitset<float, sizeof(float)>(2.0f);

    BOOST_TEST(result.to_string() == "01000000000000000000000000000000");
  }
  {
    const auto result = floating_point_to_bitset<float, sizeof(float)>(2.5f);

    BOOST_TEST(result.to_string() == "01000000001000000000000000000000");
  }
  {
    const auto result = floating_point_to_bitset<float, sizeof(float)>(2.25f);

    BOOST_TEST(result.to_string() == "01000000000100000000000000000000");
  }
  {
    const auto result = floating_point_to_bitset<float, sizeof(float)>(1.0f);

    BOOST_TEST(result.to_string() == "00111111100000000000000000000000");
  }
  {
    const auto result = floating_point_to_bitset<float, sizeof(float)>(0.5f);

    BOOST_TEST(result.to_string() == "00111111000000000000000000000000");
  }
  {
    const auto result = floating_point_to_bitset<float, sizeof(float)>(0.25f);

    BOOST_TEST(result.to_string() == "00111110100000000000000000000000");
  }
  {
    const auto result = floating_point_to_bitset<float, sizeof(float)>(0.125f);

    BOOST_TEST(result.to_string() == "00111110000000000000000000000000");
  }
  {
    const auto result = floating_point_to_bitset<float, sizeof(float)>(0.0625f);

    BOOST_TEST(result.to_string() == "00111101100000000000000000000000");
  }
  {
    const auto result = floating_point_to_bitset<float, sizeof(float)>(
      0.03125f);

    BOOST_TEST(result.to_string() == "00111101000000000000000000000000");
  }
  // cf. https://www.cs.utexas.edu/users/fussell/courses/cs429h/lectures/Lecture_4-429h.pdf
  {
    const auto result = floating_point_to_bitset<float, sizeof(float)>(
      15213.0f);

    BOOST_TEST(result.to_string() == "01000110011011011011010000000000");
  }
}

// cf. https://en.cppreference.com/w/cpp/types/numeric_limits
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FloatingPointToBitSetTakesLimits)
{
  {
    // Returns the lowest finite value of the given type
    // Lowest:-3.40282e+38
    const auto result = floating_point_to_bitset<float, sizeof(float)>(
      std::numeric_limits<float>::lowest());

    BOOST_TEST(result.to_string() == "11111111011111111111111111111111");
  }
  {
    // Returns the smallest finite value of the given type
    // min:1.17549e-38
    const auto result = floating_point_to_bitset<float, sizeof(float)>(
      std::numeric_limits<float>::min());

    BOOST_TEST(result.to_string() == "00000000100000000000000000000000");
  }
  {
    // Returns the largest finite value of the given type
    // max:3.40282e+38
    const auto result = floating_point_to_bitset<float, sizeof(float)>(
      std::numeric_limits<float>::max());

    BOOST_TEST(result.to_string() == "01111111011111111111111111111111");
  }
}

BOOST_AUTO_TEST_SUITE_END() // ConversionsWithUnion_tests

BOOST_AUTO_TEST_SUITE_END() // FloatingPoint_tests
BOOST_AUTO_TEST_SUITE_END() // Utilities
