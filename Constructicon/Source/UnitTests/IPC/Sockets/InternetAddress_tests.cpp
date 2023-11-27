#include "IPC/Sockets/InternetAddress.h"
#include "Utilities/Testing/CaptureCout.h"

#include <gtest/gtest.h>
#include <iostream>

using IPC::Sockets::InternetAddress;
using IPC::Sockets::InternetSocketAddress;
using Utilities::Testing::CaptureCout;
using std::cout;

namespace GoogleUnitTests
{
namespace IPC
{
namespace Sockets
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(InternetSocketAddressTests, DefaultConstructs)
{
  CaptureCout capture_cout {};

  InternetSocketAddress address {};
  cout << address;

  EXPECT_EQ(
    capture_cout.local_oss_.str(),
    "sin_family: 2, sin_port: 0, s_addr: 0\n");
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(InternetAddressTests, DefaultConstructs)
{
  CaptureCout capture_cout {};

  InternetAddress address {};
  cout << address;

  EXPECT_EQ(
    capture_cout.local_oss_.str(),
    "sin_family: 2, sin_port: 0, s_addr: 0\n");
}

} // namespace Sockets

} // namespace IPC

} // namespace GoogleUnitTests