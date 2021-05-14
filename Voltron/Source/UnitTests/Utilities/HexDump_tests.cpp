#include "Utilities/HexDump.h"

#include "Tools/CaptureCout.h"

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <string>

using Tools::CaptureCoutFixture;
using Utilities::hex_dump;
using std::cout;
using std::string;

BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(HexDump_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(DemonstrateHexDumpPrintOut,
	CaptureCoutFixture)
{
	// 35 chars.
	char my_string[] {"a char string greater than 16 chars"};

	hex_dump(my_string, sizeof(my_string), cout);

	restore_cout();

	// \ref https://stackoverflow.com/questions/7775991/how-to-get-hexdump-of-a-structure-data
	// Validate manually that the hex values match those in the original example.

	string expected {
		"0000 : a char string gr 61 20 63 68 61 72 20 73 74 72 69 6E 67 20 67 72 \n"};
	expected +=
		"0010 : eater than 16 ch 65 61 74 65 72 20 74 68 61 6E 20 31 36 20 63 68 \n";
	expected += "0020 : ars.             61 72 73 00 \n";

  BOOST_TEST(local_oss_.str() == expected);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
/*
BOOST_AUTO_TEST_CASE(DemonstrateHexDumpCout)
{
	char my_string[] {"a char string greater than 16 chars"};

	cout << "\n hex dump : " << sizeof(my_string) / sizeof(char) << "\n";

	hex_dump(my_string, 34, cout);
}
*/

BOOST_AUTO_TEST_SUITE_END() // HexDump_tests
BOOST_AUTO_TEST_SUITE_END() // Utilities
