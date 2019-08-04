//------------------------------------------------------------------------------
// \file NarrowCast_test.cpp
//------------------------------------------------------------------------------

#include <boost/test/unit_test.hpp>
#include <iostream>

#include "Utilities/NarrowCast.h"

using Utilities::narrow_cast;

BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(NarrowCast_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(NarrowCast)
{
  std::cout << "\n NarrowCastIsFun\n";

  BOOST_TEST(true);
}

BOOST_AUTO_TEST_SUITE_END() // NarrowCast_tests
BOOST_AUTO_TEST_SUITE_END() // Utilities
