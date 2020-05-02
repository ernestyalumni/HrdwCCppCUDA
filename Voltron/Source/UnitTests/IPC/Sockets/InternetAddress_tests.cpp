//------------------------------------------------------------------------------
/// \file InternetAddress_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
//------------------------------------------------------------------------------
#include "IPC/Sockets/InternetAddress.h"

#include <boost/test/unit_test.hpp>

using IPC::Sockets::InternetAddress;

BOOST_AUTO_TEST_SUITE(IPC)
BOOST_AUTO_TEST_SUITE(Sockets)
BOOST_AUTO_TEST_SUITE(InternetAddress_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DomainsConvertToIntValues)
{
  BOOST_TEST(true);
}

BOOST_AUTO_TEST_SUITE_END() // InternetAddress_tests
BOOST_AUTO_TEST_SUITE_END() // Sockets
BOOST_AUTO_TEST_SUITE_END() // IPC