//------------------------------------------------------------------------------
// \file Enumeration_tests.cpp
//------------------------------------------------------------------------------
#include "Cpp/Utilities/TypeSupport/UnderlyingTypes.h"

#include <boost/test/unit_test.hpp>

using Cpp::Utilities::TypeSupport::get_underlying_value;

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Enumeration_tests)

enum class NameToValue : char
{
	Alpha = 'a',
	Beta = 'b',
	Gamma = 'c'
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StaticCastToEnumerationType)
{
	BOOST_TEST((static_cast<NameToValue>('a') == NameToValue::Alpha));
	BOOST_TEST((static_cast<NameToValue>('b') == NameToValue::Beta));
	BOOST_TEST((static_cast<NameToValue>('c') == NameToValue::Gamma));

	const NameToValue casted_d {static_cast<NameToValue>('d')};

	BOOST_TEST((casted_d != NameToValue::Alpha));
	BOOST_TEST((casted_d != NameToValue::Beta));
	BOOST_TEST((casted_d != NameToValue::Gamma));
}

BOOST_AUTO_TEST_SUITE_END() // Enumeration_tests
BOOST_AUTO_TEST_SUITE_END() // Cpp