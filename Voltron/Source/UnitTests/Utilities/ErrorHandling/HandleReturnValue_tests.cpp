#include "Utilities/ErrorHandling/HandleReturnValue.h"

#include "FileIO/FileFlagsModes.h"
#include "Tools/TemporaryDirectory.h"

#include <boost/test/unit_test.hpp>
#include <fcntl.h>
#include <string>

using Tools::TemporaryDirectory;
using FileIO::AccessMode;
using FileIO::to_access_mode_value;
using Utilities::ErrorHandling::HandleReturnValue;

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

  handle_return_value(open_result);

  const auto error_number = handle_return_value.error_number();

//  BOOST_TEST(handle_return_value.error_number().error_number() == 23);

}

BOOST_AUTO_TEST_SUITE_END() // HandleReturnValue_tests
BOOST_AUTO_TEST_SUITE_END() // Utilities