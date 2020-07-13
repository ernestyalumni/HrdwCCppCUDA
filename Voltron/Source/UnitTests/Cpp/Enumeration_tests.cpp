//------------------------------------------------------------------------------
// \file Enumeration_tests.cpp
//------------------------------------------------------------------------------
#include "Cpp/Numerics/BitCast.h"
#include "Cpp/Utilities/SuperBitSet.h"
#include "Cpp/Utilities/TypeSupport/UnderlyingTypes.h"

#include <boost/test/unit_test.hpp>
#include <utility>

using Cpp::Numerics::bit_cast;
using Cpp::Utilities::SuperBitSet;
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

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(EnumClassValuesCanBeComparedWithBitwiseOperations)
{
	BOOST_TEST(
		get_underlying_value(DefaultUnsignedChars::Trois) ==
			get_underlying_value(DefaultUnsignedChars::Un) |
				get_underlying_value(DefaultUnsignedChars::Deux));

	SuperBitSet<8> un_bits8 {bit_cast<uint8_t>(DefaultUnsignedChars::Un)};
	BOOST_TEST(un_bits8.to_string() == "00000001");

	SuperBitSet<8> deux_bits8 {bit_cast<uint8_t>(DefaultUnsignedChars::Deux)};
	BOOST_TEST(deux_bits8.to_string() == "00000010");

	SuperBitSet<8> trois_bits8 {bit_cast<uint8_t>(DefaultUnsignedChars::Trois)};
	BOOST_TEST(trois_bits8.to_string() == "00000011");

	const int32_t un_et_trois {
		get_underlying_value(DefaultUnsignedChars::Un) &
			get_underlying_value(DefaultUnsignedChars::Trois)};

	SuperBitSet<32> un_et_trois_bits32 {bit_cast<uint32_t>(un_et_trois)};
	BOOST_TEST(un_et_trois_bits32.to_string() ==
		"00000000000000000000000000000001");
	
	/* // Narrowing conversion
	const bool est_un_et_trois {
		get_underlying_value(DefaultUnsignedChars::Un) &
			get_underlying_value(DefaultUnsignedChars::Trois)};
	*/

	BOOST_TEST(static_cast<bool>(un_et_trois));

	const int32_t deux_et_trois {
		get_underlying_value(DefaultUnsignedChars::Deux) &
			get_underlying_value(DefaultUnsignedChars::Trois)};

	SuperBitSet<32> deux_et_trois_bits32 {bit_cast<uint32_t>(deux_et_trois)};
	BOOST_TEST(deux_et_trois_bits32.to_string() ==
		"00000000000000000000000000000010");

	BOOST_TEST(static_cast<bool>(deux_et_trois));

	const int32_t zero_et_trois {
		get_underlying_value(DefaultUnsignedChars::Zero) &
			get_underlying_value(DefaultUnsignedChars::Trois)};
	BOOST_TEST(!static_cast<bool>(zero_et_trois));

	const int32_t quatre_et_trois {
		get_underlying_value(DefaultUnsignedChars::Quatre) &
			get_underlying_value(DefaultUnsignedChars::Trois)};
	BOOST_TEST(!static_cast<bool>(quatre_et_trois));
}

BOOST_AUTO_TEST_SUITE_END() // Enumeration_tests
BOOST_AUTO_TEST_SUITE_END() // Cpp