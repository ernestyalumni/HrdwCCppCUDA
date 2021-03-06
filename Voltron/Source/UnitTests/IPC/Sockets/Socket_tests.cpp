//------------------------------------------------------------------------------
/// \file Socket_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
//------------------------------------------------------------------------------
#include "IPC/Sockets/Socket.h"

#include "IPC/Sockets/ParameterFamilies.h"
#include "UnitTests/Tools/Contains.h"

#include <boost/test/unit_test.hpp>
#include <sys/socket.h>
#include <system_error>

using IPC::Sockets::Domains;
using IPC::Sockets::Socket;
using IPC::Sockets::Types;
using UnitTests::Tools::error_contains;

BOOST_AUTO_TEST_SUITE(IPC)
BOOST_AUTO_TEST_SUITE(Sockets)
BOOST_AUTO_TEST_SUITE(Socket_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SocketConstructsWithSpecialEnumClasses)
{
  {
    const Socket socket {Domains::ipv4, Types::datagram};
    BOOST_TEST(socket.domain() == AF_INET);
    BOOST_TEST(socket.type() == SOCK_DGRAM);

    // cf. https://www.cs.rutgers.edu/~pxk/417/notes/sockets/demo-udp-01.html
    // std::cout << "\ncreated socket: descriptor: " << socket.fd() << '\n';
  }
  {
    const Socket socket {Domains::unix_, Types::stream};
    BOOST_TEST(socket.domain() == AF_UNIX);
    BOOST_TEST(socket.type() == SOCK_STREAM);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SocketThrowsForUnsupportedProtocols)
{
	{
    // TODO: Restore this unit test once error handling settled.
    /*
		BOOST_CHECK_EXCEPTION(
			Socket(Domains::ipv4, Types::raw),
			std::system_error,
			error_contains("93"));
    */
	}
}

BOOST_AUTO_TEST_SUITE_END() // Socket_tests
BOOST_AUTO_TEST_SUITE_END() // Sockets
BOOST_AUTO_TEST_SUITE_END() // IPC