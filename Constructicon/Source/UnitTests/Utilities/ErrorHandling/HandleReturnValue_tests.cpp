#include "IPC/Sockets/ParameterFamilies.h"
#include "Utilities/ErrorHandling/HandleReturnValue.h"
#include "Utilities/TypeSupport/GetUnderlyingValue.h"

#include <gtest/gtest.h>
#include <sys/socket.h>

using IPC::Sockets::Domain;
using IPC::Sockets::Type;
using Utilities::ErrorHandling::HandleReturnValue;
using Utilities::TypeSupport::get_underlying_value;

namespace GoogleUnitTests
{
namespace Utilities
{
namespace ErrorHandling
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(HandleReturnValueTests, DefaultConstructs)
{
  HandleReturnValue handle {};

  EXPECT_EQ(handle.error_number(), 0);
  EXPECT_EQ(handle.as_string(), "Success");
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(HandleReturnValueTests, HandlesReturnValues)
{
  HandleReturnValue handle {};

  handle(::socket(
    get_underlying_value(Domain::maximum) + 1,
    get_underlying_value(Type::stream),
    0));

  EXPECT_EQ(handle.error_number(), 97);
  EXPECT_EQ(handle.as_string(), "Address family not supported by protocol");
  EXPECT_EQ(handle.return_value(), -1);
  // https://man7.org/linux/man-pages/man2/socket.2.html
  // Errors - EAFNOSUPPORT - The implementation doesn't support the specified
  // address family.
  EXPECT_EQ(handle.error_number(), EAFNOSUPPORT);

  errno = 0;
}

} // namespace Errorhandling
} // namespace Utilities
} // namespace GoogleUnitTests