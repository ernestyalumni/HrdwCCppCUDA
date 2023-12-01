#include "Utilities/ErrorHandling/GetErrorNumber.h"

#include <cmath>
#include <gtest/gtest.h>
#include <system_error>
#include <thread>

using Utilities::ErrorHandling::GetErrorNumber;

namespace GoogleUnitTests
{
namespace Utilities
{
namespace ErrorHandling
{

// See https://en.cppreference.com/w/cpp/error/system_error
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetErrorNumberTests, ConstructsFromSystemError)
{
  try
  {
    std::thread().detach(); // attempt to detach a non-thread
  }
  catch (const std::system_error& err)
  {
    GetErrorNumber get_error_number {err};

    EXPECT_EQ(get_error_number.error_number(), 22);
    EXPECT_EQ(get_error_number.as_string(), "Invalid argument");
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetErrorNumberTests, DefaultConstructorGetsLatestErrnoValue)
{
  GetErrorNumber get_error_number_0 {};

  EXPECT_EQ(get_error_number_0.error_number(), 0);
  EXPECT_EQ(get_error_number_0.as_string(), "Success");

  std::log(-1.0);

  EXPECT_EQ(get_error_number_0.error_number(), 0);

  GetErrorNumber get_error_number_1 {};

  EXPECT_EQ(get_error_number_1.error_number(), 33);
  EXPECT_EQ(get_error_number_1.as_string(), "Numerical argument out of domain");

  errno = 0;

  GetErrorNumber get_error_number_2 {};

  EXPECT_EQ(get_error_number_2.error_number(), 0);
  EXPECT_EQ(get_error_number_2.as_string(), "Success");

}

} // namespace Errorhandling
} // namespace Utilities
} // namespace GoogleUnitTests