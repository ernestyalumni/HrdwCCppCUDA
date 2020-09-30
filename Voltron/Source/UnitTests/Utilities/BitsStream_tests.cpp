//------------------------------------------------------------------------------
// \file BitsStream_test.cpp
//------------------------------------------------------------------------------
#include "Utilities/BitsStream.h"

#include "Cpp/Utilities/SuperBitSet.h"

#include <boost/test/unit_test.hpp>
#include <cmath>
#include <limits>
#include <string>
// https://stackoverflow.com/questions/20731/how-do-you-clear-a-stringstream-variable

using Cpp::Utilities::SuperBitSet;
using Utilities::BitsStream;

BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(BitsStream_tests)

// cf. https://stackoverflow.com/questions/34588650/uint128-t-does-not-name-a-type
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(__Uint128_tExists)
{
  BOOST_TEST(sizeof(__uint128_t) == 16);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateMinus0In1sComplement)
{
  SuperBitSet<16> bits {BitsStream::minus_0_1s_complement};
  BOOST_TEST(bits.to_string() == "1111111111111111");
  BOOST_TEST(bits.to_ulong() == 65535);

}


BOOST_AUTO_TEST_SUITE_END() // BitsStream_tests
BOOST_AUTO_TEST_SUITE_END() // Utilities
