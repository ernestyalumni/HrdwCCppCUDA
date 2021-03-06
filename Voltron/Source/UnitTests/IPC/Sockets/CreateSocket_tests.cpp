//------------------------------------------------------------------------------
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
//------------------------------------------------------------------------------
#include "IPC/Sockets/CreateSocket.h"

#include "IPC/Sockets/ParameterFamilies.h"

#include <boost/test/unit_test.hpp>
#include <sys/socket.h>

using IPC::Sockets::CreateSocket;
using IPC::Sockets::Domain;
using IPC::Sockets::Type;

BOOST_AUTO_TEST_SUITE(IPC)
BOOST_AUTO_TEST_SUITE(Sockets)
BOOST_AUTO_TEST_SUITE(CreateSocket_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithEnumerationConstants)
{
  {
    CreateSocket create_socket {Domain::ipv4, Type::datagram};
    BOOST_TEST(create_socket.type_value() == SOCK_DGRAM);
  }

}

BOOST_AUTO_TEST_SUITE_END() // CreateSocket_tests
BOOST_AUTO_TEST_SUITE_END() // Sockets
BOOST_AUTO_TEST_SUITE_END() // IPC