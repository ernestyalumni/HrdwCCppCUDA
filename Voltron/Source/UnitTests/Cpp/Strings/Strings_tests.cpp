//------------------------------------------------------------------------------
/// \file Strings_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \details Tests of string_view, class template describing object that can
///   refer to a constant contiguous sequence of char-like objects with first
///   element of sequence at position 0.
/// \ref https://en.cppreference.com/w/cpp/string/basic_string_view
/// 
//------------------------------------------------------------------------------
#include <array>
#include <boost/test/unit_test.hpp>
#include <cctype>
#include <cstdio> // std::snprintf;
#include <iostream>
#include <string>
#include <string_view>

using std::array;
using std::isprint;
using std::string;
using std::snprintf;

template <typename T>
string printable(const T& string_text)
{
	string result {};
	result.reserve(string_text.size() * 3 + 2); // Worse case, all not printable +
	// final "}}"

	bool is_printable {true};

	// First index is for is_printable.
	// Second index is for was_printable.
	static const char* format_string[2][2] {
		{".%02hhx", "{{%02hhx"},
		{"}}%c", "%c"}
	};

	for (auto c : string_text)
	{
		const bool was_printable {is_printable};

		is_printable = isprint(static_cast<unsigned char>(c));

		// cf. https://stackoverflow.com/questions/39548254/does-stdarray-guarantee-allocation-on-the-stack-only
		// std::array is on the stack.
		// C++ has no concept of stack or heap.
		// It has free store. new accesses the free store. Variables "on the stack"
		// go to automatic storage.
		// In practice, in order to allocate on free store, you have to risk out of
		// memory exception.

		array<char, 4> buffer; // "?xx\0"

		snprintf(
			buffer.data(),
			buffer.size(),
			format_string[is_printable][was_printable],
			c);

		result += buffer.data();
	}

	if (!is_printable)
	{
		result += "}}";
	}

	return result;
}

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Strings)
BOOST_AUTO_TEST_SUITE(Strings_tests)

BOOST_AUTO_TEST_SUITE(CcType_tests)

//------------------------------------------------------------------------------
/// \ref https://en.cppreference.com/w/cpp/string/byte/isprint
/// int isprint(int ch)
/// \details Checks if ch is a printable character as classified by currently
/// installed C locale.
/// In default, following are printable,
/// * digits,
/// * uppercase letters, lowercase letters
/// * punctuation characters (!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~) 
/// * space ( )
/// behavior undefined if value of ch isn't representable as unsigned char and
/// is not equal to EOF.
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IsprintDistinguishesPrintableCharacters)
{
	const string affirmation {"Please believe it, if the mind can conceive it,"};

	for (auto c : affirmation)
	{
		BOOST_TEST(isprint(c));
	}

	const char test_new_line {'\n'};

	BOOST_TEST(!isprint(test_new_line));

	const string test_hex_1 {"\x01"};

	BOOST_TEST(test_hex_1.length() == 1);
	BOOST_TEST(!isprint(*test_hex_1.begin()));

	const string test_hex_12 {"\x12"};

	BOOST_TEST(test_hex_12.length() == 1);
	BOOST_TEST(!isprint(*test_hex_12.begin()));

	const string test_back_slash {"\\"};
	BOOST_TEST(test_back_slash.length() == 1);
	BOOST_TEST(isprint(*test_back_slash.begin()));

	const string broken_hex_string {"\x1\x01\x2\x02"};

	BOOST_TEST(broken_hex_string.length() == 4);

	for (char c : broken_hex_string)
	{
		BOOST_TEST(!isprint(c));
	}

	const string printable_values_in_hex_format {"\x42\x69"};

	BOOST_TEST(printable_values_in_hex_format.length() == 2);

	for (char c : printable_values_in_hex_format)
	{
		BOOST_TEST(isprint(c));
	}

	const string null_terminated_as_hex {"\x0\x42\x69"};
	BOOST_TEST(null_terminated_as_hex.length() == 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IsprintUsedToProcessString)
{
	{
		const string affirmation {"Then the man can achieve it."};
		const auto result = printable(affirmation);

		BOOST_TEST(result == affirmation);
	}
	{
		const string test_hex_02 {"\x02"};
		BOOST_TEST(test_hex_02.length() == 1);
		const auto result = printable(test_hex_02);

		BOOST_TEST(result == "{{0}}");
	}
	{
		const string test_hex_12 {"\x12"};
		BOOST_TEST(test_hex_12.length() == 1);
		const auto result = printable(test_hex_12);

		BOOST_TEST(result == "{{1}}");
	}

}

BOOST_AUTO_TEST_SUITE_END() // CcType_tests

BOOST_AUTO_TEST_SUITE_END() // Strings_tests
BOOST_AUTO_TEST_SUITE_END() // Strings
BOOST_AUTO_TEST_SUITE_END() // Cpp