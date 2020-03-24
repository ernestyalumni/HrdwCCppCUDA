//------------------------------------------------------------------------------
// \file Max_tests.cpp
//------------------------------------------------------------------------------
#include "Cpp/Templates/Basics/Max.h"

#include <boost/test/unit_test.hpp>

using Cpp::Templates::Basics::max;

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Templates)
BOOST_AUTO_TEST_SUITE(Basics)
BOOST_AUTO_TEST_SUITE(Max_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateMax)
{
	{
		// Works for ints
		const int a {4};
		const int b {5};
		BOOST_TEST(max(a, b) == b);
	}

	BOOST_TEST(true);
}


BOOST_AUTO_TEST_SUITE_END() // Max_tests
BOOST_AUTO_TEST_SUITE_END() // Basics
BOOST_AUTO_TEST_SUITE_END() // Templates
BOOST_AUTO_TEST_SUITE_END() // Cpp