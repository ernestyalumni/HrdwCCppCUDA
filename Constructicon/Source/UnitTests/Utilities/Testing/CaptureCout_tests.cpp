#include "Utilities/Testing/CaptureCout.h"

#include <iostream>
#include <gtest/gtest.h>
#include <sstream>
#include <string>

using Utilities::Testing::CaptureCout;
using Utilities::Testing::capture_cout;
using std::cout;
using std::ostringstream;

namespace GoogleUnitTests
{
namespace Utilities
{
namespace Testing
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CaptureCoutTests, CaptureCoutAcceptsLocalOStringStream)
{
  ostringstream local_oss;

  auto cout_buffer_ptr = capture_cout(local_oss);

  cout << "some message";

  cout.rdbuf(cout_buffer_ptr);

  const std::string expected_message {"some message"};
  EXPECT_EQ(local_oss.str(), expected_message);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CaptureCoutTests, DefaultConstructs)
{
  CaptureCout capture_cout {};
  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CaptureCoutTests, CaptureLocallyUponConstructionCapturesLocally)
{
  CaptureCout capture_cout {};
  cout << "\n Testing Testing \n";

  EXPECT_EQ(capture_cout.local_oss_.str(), "\n Testing Testing \n");
}

} // namespace Testing
} // namespace Utilities
} // namespace GoogleUnitTests
