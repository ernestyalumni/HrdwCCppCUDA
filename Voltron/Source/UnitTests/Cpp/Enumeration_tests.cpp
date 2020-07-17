//------------------------------------------------------------------------------
// \file Enumeration_tests.cpp
//------------------------------------------------------------------------------
#include "Cpp/ClassAsEnum.h"
#include "Cpp/Numerics/BitCast.h"
#include "Cpp/Utilities/SuperBitSet.h"
#include "Cpp/Utilities/TypeSupport/UnderlyingTypes.h"

#include <boost/test/unit_test.hpp>
#include <utility>

using Cpp::ClassAsEnum;
using Cpp::ClassAsEnumClass;
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


inline bool operator&(
	const DefaultUnsignedChars x,
	const DefaultUnsignedChars y)
{
	return static_cast<bool>(get_underlying_value(x) & get_underlying_value(y));
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

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(EnumClassValuesAndBitwiseAndOperator)
{
	// error: no match for ‘operator&’ (operand types are ‘Cpp::Enumeration_tests::DefaultUnsignedChars’ and ‘Cpp::Enumeration_tests::DefaultUnsignedChars’)

	const auto underlying_un = get_underlying_value(DefaultUnsignedChars::Un);
	BOOST_TEST(sizeof(underlying_un) == 1);

	// cf. https://stackoverflow.com/questions/17489696/bitwise-shift-promotes-unsigned-char-to-int
	// Because of the standard, language promotes anything smaller than int to int
	// in bitwise operation.
	const auto deux_et_trois =
		get_underlying_value(DefaultUnsignedChars::Deux) &
			get_underlying_value(DefaultUnsignedChars::Trois);

	BOOST_TEST(sizeof(deux_et_trois) > sizeof(unsigned char));
	BOOST_TEST(sizeof(deux_et_trois) == sizeof(int));

	BOOST_TEST(!(DefaultUnsignedChars::Zero & DefaultUnsignedChars::Un));
	BOOST_TEST(!(DefaultUnsignedChars::Zero & DefaultUnsignedChars::Deux));
	BOOST_TEST(!(DefaultUnsignedChars::Zero & DefaultUnsignedChars::Trois));
	BOOST_TEST(!(DefaultUnsignedChars::Zero & DefaultUnsignedChars::Quatre));
	BOOST_TEST(!(DefaultUnsignedChars::Un & DefaultUnsignedChars::Deux));
	BOOST_TEST((DefaultUnsignedChars::Un & DefaultUnsignedChars::Trois));
	BOOST_TEST(!(DefaultUnsignedChars::Un & DefaultUnsignedChars::Quatre));
	BOOST_TEST((DefaultUnsignedChars::Deux & DefaultUnsignedChars::Trois));
	BOOST_TEST(!(DefaultUnsignedChars::Deux & DefaultUnsignedChars::Quatre));
	BOOST_TEST(!(DefaultUnsignedChars::Trois & DefaultUnsignedChars::Quatre));
}

BOOST_AUTO_TEST_SUITE_END() // Enumeration_tests

BOOST_AUTO_TEST_SUITE(ClassAsEnum_tests)

enum AltroEnum : unsigned char
{
	Zero,
	Uno,
	Due,
	Tre,
	Quattro
};

enum class DalsiEnumClass : unsigned char
{
	Nula,
	Jeden,
	Dva,
	Tri,
	Ctyri
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ClassAsEnumBehavesAsEnum)
{
	const ClassAsEnum f0 {ClassAsEnum::Zero};
	const ClassAsEnum f1 {ClassAsEnum::Un};

	BOOST_TEST(f0.est_zero());
	BOOST_TEST(f1.est_un_o_trois());

	BOOST_TEST(f0 == ClassAsEnum::Zero);
	BOOST_TEST(f1 == ClassAsEnum::Un);

}

#ifdef FORCE_COMPILE_WARNING
// Compilation warning:
// warning: comparison between ‘enum Cpp::ClassAsEnum::Valeur’ and ‘enum Cpp::ClassAsEnum_tests::AltroEnum’
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ClassAsEnumComparesWithOtherEnumValues)
{
	ClassAsEnum::Zero == AltroEnum::Zero;
	ClassAsEnum::Un == AltroEnum::Uno;
}

#endif // FORCE_COMPILE_WARNING

#ifdef FORCE_COMPILE_ERROR
// Compilation error: no match for ‘operator==’
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ClassAsEnumDoesNotCompareWithOtherEnumClassValues)
{
	ClassAsEnum::Zero == DalsiEnumClass::Nula;
	ClassAsEnum::Un == DalsiEnumClass::Jeden;
}
#endif // FORCE_COMPILE_ERROR

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ClassAsEnumClassConstructs)
{
	const ClassAsEnumClass f0 {ClassAsEnumClass::Valeur::Zero};
	const ClassAsEnumClass f1 {ClassAsEnumClass::Valeur::Un};

	BOOST_TEST(f0.est_zero());
	BOOST_TEST(f1.est_un_o_trois());
	BOOST_TEST(!f1.est_deux_o_trois());
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ClassAsEnumClassComparesWithOtherEnumValues)
{
	const ClassAsEnumClass f0 {ClassAsEnumClass::Valeur::Zero};
	const ClassAsEnumClass f1 {ClassAsEnumClass::Valeur::Un};

	//f0 == AltroEnum::Zero;
}

BOOST_AUTO_TEST_SUITE_END() // ClassAsEnum_tests


BOOST_AUTO_TEST_SUITE_END() // Cpp