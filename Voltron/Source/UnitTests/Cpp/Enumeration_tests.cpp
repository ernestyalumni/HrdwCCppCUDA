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

enum class DefaultUnsignedChars : unsigned char
{
	Zero,
	Un,
	Deux,
	Trois,
	Quatre
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

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(UnsignedCharValuesDefaultToNumbers)
{
	BOOST_TEST((get_underlying_value(DefaultUnsignedChars::Zero) == 0));

	BOOST_TEST(
		static_cast<int>(get_underlying_value(DefaultUnsignedChars::Zero)) == 0);

	//const unsigned int x {0};

	//std::cout << "\n0 as char: " << static_cast<char>(x) << "\n";

	char message[] {0, 1, 2, 3};

	BOOST_TEST((get_underlying_value(DefaultUnsignedChars::Zero) == message[0]));
	BOOST_TEST((get_underlying_value(DefaultUnsignedChars::Un) == message[1]));
	BOOST_TEST((get_underlying_value(DefaultUnsignedChars::Deux) == message[2]));
	BOOST_TEST((get_underlying_value(DefaultUnsignedChars::Trois) == message[3]));

	unsigned char message2[] {0, 1, 2, 3};

	BOOST_TEST((get_underlying_value(DefaultUnsignedChars::Zero) == message2[0]));
	BOOST_TEST((get_underlying_value(DefaultUnsignedChars::Un) == message2[1]));
	BOOST_TEST((get_underlying_value(DefaultUnsignedChars::Deux) == message2[2]));
	BOOST_TEST(
		(get_underlying_value(DefaultUnsignedChars::Trois) == message2[3]));

}

BOOST_AUTO_TEST_SUITE_END() // Enumeration_tests
BOOST_AUTO_TEST_SUITE_END() // Cpp