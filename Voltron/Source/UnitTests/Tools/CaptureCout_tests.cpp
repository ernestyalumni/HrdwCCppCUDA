//------------------------------------------------------------------------------
/// \file CaptureCout_tests.cpp
//------------------------------------------------------------------------------
#include "Tools/CaptureCout.h"

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <sstream> // std::ostringstream

using Tools::CaptureCoutFixture;
using Tools::capture_cout;
using std::cout;
using std::ostringstream;

BOOST_AUTO_TEST_SUITE(Tools)
BOOST_AUTO_TEST_SUITE(CaptureCout_tests)
BOOST_AUTO_TEST_SUITE(Capture_Cout_tests)

// cf. https://en.cppreference.com/w/cpp/io/basic_ios/rdbuf
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CaptureCoutReturnsNewAndOldStreams)
{
  auto result = capture_cout();

  // Now std::cout work with "local" buffer.
  // You don't see this message.
  cout << "some message";

  // Go back to old buffer.
  cout.rdbuf(result.second);

  // Uncomment to see this message.
  //cout << "back to default buffer\n";

  // TODO: Debug this.
  //cout << " result: " << result.first.get().str() << "end\n";

  //BOOST_TEST(result.first.get().str() == "some message");
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CaptureCoutAcceptsLocalOStringStream)
{
  ostringstream local_oss;

  //auto cout_buffer_ptr = capture_cout(local_oss);

  //cout << "some message";
}

BOOST_AUTO_TEST_SUITE_END() // Capture_Cout_tests

BOOST_AUTO_TEST_SUITE(CaptureCoutFixture_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
  CaptureCoutFixture capture_cout {};
  BOOST_TEST(true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CaptureLocallyUponConstructionCapturesLocally)
{
  CaptureCoutFixture capture_cout {};
  cout << "\n Testing Testing \n";

  BOOST_TEST(capture_cout.local_oss_.str() == "\n Testing Testing \n");
}


BOOST_AUTO_TEST_SUITE_END() // CaptureCoutFixture_tests

BOOST_AUTO_TEST_SUITE_END() // CaptureCout_tests
BOOST_AUTO_TEST_SUITE_END() // Tools