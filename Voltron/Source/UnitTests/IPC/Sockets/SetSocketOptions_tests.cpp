//------------------------------------------------------------------------------
/// \file SetSocketOptions_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
//------------------------------------------------------------------------------
#include "IPC/Sockets/SetSocketOptions.h"

#include "IPC/Sockets/InternetAddress.h"
#include "IPC/Sockets/ParameterFamilies.h"
#include "IPC/Sockets/Socket.h"

#include <boost/test/unit_test.hpp>

using IPC::Sockets::Domains;
using IPC::Sockets::InternetSocketAddress;
using IPC::Sockets::SetReusableAddressAndPort;
using IPC::Sockets::SetReusableSocketAddress;
using IPC::Sockets::SetSocketOptions;
using IPC::Sockets::Socket;
using IPC::Sockets::Types;

BOOST_AUTO_TEST_SUITE(IPC)
BOOST_AUTO_TEST_SUITE(Sockets)
BOOST_AUTO_TEST_SUITE(SetSocketOptions_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SetSocketOptions)
{
  {
    Socket socket {Domains::ipv4, Types::datagram};
    BOOST_TEST_REQUIRE(socket.domain() == AF_INET);
    BOOST_TEST_REQUIRE(socket.type() == SOCK_DGRAM);

    const InternetSocketAddress internet_socket_address {0};

    const auto result = SetReusableSocketAddress()(socket);
    BOOST_TEST(!static_cast<bool>(result));
  }
}

BOOST_AUTO_TEST_SUITE_END() // SetSocketOptions_tests
BOOST_AUTO_TEST_SUITE_END() // Sockets
BOOST_AUTO_TEST_SUITE_END() // IPC