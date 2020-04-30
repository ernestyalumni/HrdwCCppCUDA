//------------------------------------------------------------------------------
/// \file ParameterFamilies_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
//------------------------------------------------------------------------------
#include "IPC/Sockets/ParameterFamilies.h"

#include "Cpp/Utilities/TypeSupport/UnderlyingTypes.h"

#include <boost/test/unit_test.hpp>
#include <sys/socket.h>

using Cpp::Utilities::TypeSupport::get_underlying_value;
using IPC::Sockets::Domains;
using IPC::Sockets::Types;

BOOST_AUTO_TEST_SUITE(IPC)
BOOST_AUTO_TEST_SUITE(Sockets)
BOOST_AUTO_TEST_SUITE(ParameterFamilies_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DomainsConvertToIntValues)
{
  {
    BOOST_TEST(get_underlying_value(Domains::unix_) == AF_UNIX);
  }
  {
    Domains domain_family {Domains::local};
    BOOST_TEST(get_underlying_value(domain_family) == AF_LOCAL);
  }
  {
    const Domains domain_family {Domains::ipv4};
    BOOST_TEST(get_underlying_value(domain_family) == AF_INET);
  }
}

BOOST_AUTO_TEST_SUITE_END() // ParameterFamiles_tests
BOOST_AUTO_TEST_SUITE_END() // Sockets
BOOST_AUTO_TEST_SUITE_END() // IPC