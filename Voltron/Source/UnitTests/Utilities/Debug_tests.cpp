#include "Tools/CaptureCout.h"
#include "Utilities/Debug.h"

#include <boost/test/unit_test.hpp>

using Tools::CaptureCoutFixture;

BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(Debug_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DebugPrintsForIntVariable)
{
  CaptureCoutFixture capture_cout {};

  int x {16};

  DEBUG(x);

  BOOST_TEST(capture_cout.local_oss_.str() == "x 16\n");
}

BOOST_AUTO_TEST_SUITE_END() // Debug_tests
BOOST_AUTO_TEST_SUITE_END() // Utilities
