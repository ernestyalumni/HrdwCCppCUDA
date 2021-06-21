#include "Utilities/SplitString.h"

#include <boost/test/unit_test.hpp>
#include <sstream> // std::istringstream
#include <string>
#include <vector>

using Utilities::WordsDelimitedByCommas;
using Utilities::split_string_by_iterator;
using Utilities::split_string_by_overload;
using std::string;
using std::vector;

BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(SplitString_tests)

BOOST_AUTO_TEST_SUITE(SplitStringByIterator_tests)

std::string text {"Let me split this into words"};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SplitsStringSeparatedBySpace)
{
	const vector<string> result {split_string_by_iterator(text)};

	BOOST_TEST(result[0] == "Let");
	BOOST_TEST(result[1] == "me");
	BOOST_TEST(result[2] == "split");
	BOOST_TEST(result[3] == "this");
	BOOST_TEST(result[4] == "into");
	BOOST_TEST(result[5] == "words");
}

BOOST_AUTO_TEST_SUITE_END() // SplitStringByIterator_tests

BOOST_AUTO_TEST_SUITE(SplitStringByOverloadingInputOperator_tests)

std::string text {"Let,me,split,this,into,words"};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SplitsCommaDelimitedString)
{
	const vector<string> result {split_string_by_overload(text)};

	BOOST_TEST(result[0] == "Let");
	BOOST_TEST(result[1] == "me");
	BOOST_TEST(result[2] == "split");
	BOOST_TEST(result[3] == "this");
	BOOST_TEST(result[4] == "into");
	BOOST_TEST(result[5] == "words");
}

BOOST_AUTO_TEST_SUITE_END() // SplitStringByOverloadingInputOperator_tests

BOOST_AUTO_TEST_SUITE_END() // SplitString_tests
BOOST_AUTO_TEST_SUITE_END() // Utilities