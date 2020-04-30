//------------------------------------------------------------------------------
/// \file Socket_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
//------------------------------------------------------------------------------
#include "IPC/Sockets/Socket.h"

#include "IPC/Sockets/ParameterFamilies.h"

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <sys/socket.h>

using IPC::Sockets::Domains;
using IPC::Sockets::Socket;
using IPC::Sockets::Types;

BOOST_AUTO_TEST_SUITE(IPC)
BOOST_AUTO_TEST_SUITE(Sockets)
BOOST_AUTO_TEST_SUITE(ParameterFamilies_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SocketConstructsWithSpecialEnumClasses)
{
  {
    const Socket socket {Domains::ipv4, Types::datagram};
    BOOST_TEST(socket.domain() == AF_INET);
    BOOST_TEST(socket.type() == SOCK_DGRAM);

    // cf. https://www.cs.rutgers.edu/~pxk/417/notes/sockets/demo-udp-01.html
    std::cout << "\ncreated socket: descriptor: " << socket.fd() << '\n';
  }
}

BOOST_AUTO_TEST_SUITE_END() // ParameterFamiles_tests
BOOST_AUTO_TEST_SUITE_END() // Sockets
BOOST_AUTO_TEST_SUITE_END() // IPC