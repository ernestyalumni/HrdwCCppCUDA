//------------------------------------------------------------------------------
/// \file Bind_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
//------------------------------------------------------------------------------
#include "IPC/Sockets/Bind.h"

#include "IPC/Sockets/InternetAddress.h"
#include "IPC/Sockets/ParameterFamilies.h"
#include "IPC/Sockets/Socket.h"

#include <boost/test/unit_test.hpp>

using IPC::Sockets::Domains;
using IPC::Sockets::Socket;
using IPC::Sockets::Bind;
using IPC::Sockets::InternetSocketAddress;
using IPC::Sockets::Types;

BOOST_AUTO_TEST_SUITE(IPC)
BOOST_AUTO_TEST_SUITE(Sockets)
BOOST_AUTO_TEST_SUITE(Bind_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BindBindsSocketToAnyAvailablePort)
{
  {
    Socket socket {Domains::ipv4, Types::datagram};
    BOOST_TEST_REQUIRE(socket.domain() == AF_INET);
    BOOST_TEST_REQUIRE(socket.type() == SOCK_DGRAM);

    const InternetSocketAddress internet_socket_address {0};

    Bind bind_f {internet_socket_address};
    const auto bind_result = bind_f(socket);

    BOOST_TEST(!static_cast<bool>(bind_result));
  }
}

BOOST_AUTO_TEST_SUITE_END() // Bind_tests
BOOST_AUTO_TEST_SUITE_END() // Sockets
BOOST_AUTO_TEST_SUITE_END() // IPC