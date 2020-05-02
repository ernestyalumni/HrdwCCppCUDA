//------------------------------------------------------------------------------
/// \file Listen_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
//------------------------------------------------------------------------------
#include "IPC/Sockets/Listen.h"

#include "IPC/Sockets/Bind.h"
#include "IPC/Sockets/InternetAddress.h"
#include "IPC/Sockets/ParameterFamilies.h"
#include "IPC/Sockets/Socket.h"

#include <boost/test/unit_test.hpp>

using IPC::Sockets::Bind;
using IPC::Sockets::Domains;
using IPC::Sockets::InternetSocketAddress;
using IPC::Sockets::MakeListen;
using IPC::Sockets::Socket;
using IPC::Sockets::Types;

BOOST_AUTO_TEST_SUITE(IPC)
BOOST_AUTO_TEST_SUITE(Sockets)
BOOST_AUTO_TEST_SUITE(Listen_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BindBindsSocketToAnyAvailablePort)
{
  {
    Socket socket {Domains::ipv4, Types::stream};
    BOOST_TEST_REQUIRE(socket.domain() == AF_INET);
    BOOST_TEST_REQUIRE(socket.type() == SOCK_STREAM);

    const InternetSocketAddress internet_socket_address {21234};

    Bind bind_f {internet_socket_address};
    const auto bind_result = bind_f(socket);

    BOOST_TEST_REQUIRE(!static_cast<bool>(bind_result));

    // Setup the socket for listening with a queue length of 5.
    MakeListen listen_f {5};
    const auto listen_result = listen_f(socket);

    BOOST_TEST(!static_cast<bool>(listen_result));    
  }
}

BOOST_AUTO_TEST_SUITE_END() // Listen_tests
BOOST_AUTO_TEST_SUITE_END() // Sockets
BOOST_AUTO_TEST_SUITE_END() // IPC