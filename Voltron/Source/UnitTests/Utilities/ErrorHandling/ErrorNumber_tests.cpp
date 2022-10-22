#include "Utilities/ErrorHandling/ErrorNumber.h"

#include <boost/test/unit_test.hpp>
#include <cmath>
#include <system_error>
#include <thread>

using Utilities::ErrorHandling::ErrorCodeNumber;
using Utilities::ErrorHandling::ErrorNumber;

BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(ErrorHandling)
BOOST_AUTO_TEST_SUITE(ErrorNumber_tests)

// cf. https://en.cppreference.com/w/cpp/error/system_error
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsFromSystemError)
{
  try
  {
    std::thread().detach(); // attempt to detach a non-thread
  }
  catch (const std::system_error& err)
  {
    ErrorNumber err_number {err};

    BOOST_TEST(err_number.error_number() == 22);
    BOOST_TEST(err_number.as_string() == "Invalid argument");
  }

  BOOST_TEST(true);
}

// cf. https://en.cppreference.com/w/cpp/error/errno
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructorGetsLatestErrnoValue)
{
  ErrorNumber error_number_0 {};

  BOOST_TEST(error_number_0.error_number() == 13);
  BOOST_TEST(error_number_0.as_string() == "Permission denied");

  double not_a_number = std::log(-1.0);

  BOOST_TEST(error_number_0.error_number() == 13);

  ErrorNumber error_number_1 {};

  BOOST_TEST(error_number_1.error_number() == 33);

  BOOST_TEST(error_number_1.as_string() == "Numerical argument out of domain");
}

// TODO: Doesn't work, see reference:
// https://en.cppreference.com/w/cpp/error/error_code/error_code

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
/*
BOOST_AUTO_TEST_CASE(ConvertErrorCodeEnumerationWorksOnUserDefinedErrorCodes)
{
  ErrorNumber error_number {};

  error_number.convert_error_code_enumeration(ErrorCodeNumber::address_in_use);

  BOOST_TEST(error_number.error_number() ==
    error_number.to_error_code_value(ErrorCodeNumber::address_in_use));
}
*/

BOOST_AUTO_TEST_SUITE_END() // ErrorNumber_tests
BOOST_AUTO_TEST_SUITE_END() // ErrorHandling
BOOST_AUTO_TEST_SUITE_END() // Utilities