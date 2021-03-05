#include "Utilities/ErrorHandling/HandleError.h"

#include <boost/test/unit_test.hpp>
#include <cmath>
#include <system_error>
#include <thread>

using Utilities::ErrorHandling::HandleError;

BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(HandleError_tests)

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
    HandleError handle_error {err};

    BOOST_TEST(handle_error.error_number().error_number() == 22);
    BOOST_TEST(handle_error.error_number().as_string() == "Invalid argument");
  }

  BOOST_TEST(true);
}

// cf. https://en.cppreference.com/w/cpp/error/errno
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructorGetsLatestErrnoValue)
{
  HandleError handle_error; 

  BOOST_TEST(handle_error.error_number().error_number() == 13);
  BOOST_TEST(handle_error.error_number().as_string() == "Permission denied");

  double not_a_number = std::log(-1.0);

  BOOST_TEST(handle_error.error_number().error_number() == 13);

  handle_error();

  BOOST_TEST(handle_error.error_number().error_number() == 33);

  BOOST_TEST(handle_error.error_number().as_string() ==
    "Numerical argument out of domain");
}

BOOST_AUTO_TEST_SUITE_END() // HandleError_tests
BOOST_AUTO_TEST_SUITE_END() // Utilities