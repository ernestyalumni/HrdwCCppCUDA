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

#include <boost/test/unit_test.hpp>
#include <cctype>
#include <iostream>
#include <string>
#include <string_view>

using std::isprint;

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

}

BOOST_AUTO_TEST_SUITE_END() // CcType_tests

BOOST_AUTO_TEST_SUITE_END() // Strings_tests
BOOST_AUTO_TEST_SUITE_END() // Strings
BOOST_AUTO_TEST_SUITE_END() // Cpp