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
using Utilities::ErrorHandling::ErrorCodeNumber;
using Utilities::ErrorHandling::ErrorNumber;
using Utilities::ErrorHandling::HandleReturnValue;
using Utilities::ErrorHandling::HandleReturnValueWithOptional;
using std::string;

BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(ErrorHandling)
BOOST_AUTO_TEST_SUITE(HandleReturnValue_tests)

int open_non_existent_file(const string& directory_path)
{
  string filename {directory_path + "/Notthere.txt"};
  return ::open(
    filename.c_str(),
    to_access_mode_value(AccessMode::write_only));
}

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

BOOST_AUTO_TEST_SUITE(HandleReturnValueWithOptional_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(OperatorReturnsNullOptForValuesGreaterThanNegativeOne)
{
  HandleReturnValueWithOptional error_handle;
  auto result = error_handle(0);
  BOOST_TEST(!static_cast<bool>(result));

  result = error_handle(1);
  BOOST_TEST(!static_cast<bool>(result));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(OperatorReturnsOptionalErrorNumberForNegativeOne)
{
  HandleReturnValueWithOptional error_handle;

  TemporaryDirectory temp_dir {"Temp"};

  const int open_result {open_non_existent_file(temp_dir.path())};

  BOOST_TEST_REQUIRE(open_result == -1);

  auto result = error_handle(open_result);

  BOOST_TEST(static_cast<bool>(result));
  BOOST_TEST((*result).error_number() ==
    ErrorNumber::to_error_code_value(
      ErrorCodeNumber::no_such_file_or_directory));
}

BOOST_AUTO_TEST_SUITE_END() // HandleReturnValueWithOptional_tests

BOOST_AUTO_TEST_SUITE_END() // HandleReturnValue_tests
BOOST_AUTO_TEST_SUITE_END() // ErrorHandling
BOOST_AUTO_TEST_SUITE_END() // Utilities