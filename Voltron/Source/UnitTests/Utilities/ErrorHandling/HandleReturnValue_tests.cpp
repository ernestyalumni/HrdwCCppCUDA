#include "Utilities/ErrorHandling/HandleReturnValue.h"

#include "Utilities/ErrorHandling/ErrorNumber.h"

#include "FileIO/FileFlagsModes.h"
#include "Tools/TemporaryDirectory.h"

#include <boost/test/unit_test.hpp>
#include <fcntl.h>
#include <string>

using FileIO::AccessMode;
using FileIO::to_access_mode_value;
using Tools::TemporaryDirectory;
using Utilities::ErrorHandling::HandleReturnValue;

using Utilities::ErrorHandling::ErrorNumber;

BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(HandleReturnValue_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(OperatorDetectsErrors)
{
  TemporaryDirectory temp_dir {"Temp"};
  std::string filename {temp_dir.path() + "/Notthere.txt"};

  HandleReturnValue handle_return_value;

  const int open_result {
    ::open(filename.c_str(), to_access_mode_value(AccessMode::write_only))};

  BOOST_TEST(open_result == -1);

  ErrorNumber expected_error_number;

  handle_return_value(open_result);

  const auto error_number = handle_return_value.error_number();

	BOOST_TEST(error_number.error_number() == 2);
	BOOST_TEST(error_number.as_string() == "No such file or directory");

	BOOST_TEST(expected_error_number.error_number() == 2);
	BOOST_TEST(expected_error_number.as_string() == "No such file or directory");

}

BOOST_AUTO_TEST_SUITE_END() // HandleReturnValue_tests
BOOST_AUTO_TEST_SUITE_END() // Utilities