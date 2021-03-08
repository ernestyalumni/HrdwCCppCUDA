//------------------------------------------------------------------------------
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
//------------------------------------------------------------------------------
#include "IPC/Sockets/CreateSocket.h"

#include "IPC/Sockets/ParameterFamilies.h"

#include <boost/test/unit_test.hpp>
#include <sys/socket.h>
#include <utility>

using IPC::Sockets::CreateSocket;
using IPC::Sockets::Domain;
using IPC::Sockets::SocketFd;
using IPC::Sockets::Type;
using std::move;

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
  {
    CreateSocket create_socket {Domain::unix_, Type::stream};
    BOOST_TEST(create_socket.domain() == AF_UNIX);
    BOOST_TEST(create_socket.type_value() == SOCK_STREAM);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(OperatorCreatesSocketFd)
{
  {
    CreateSocket create_socket {Domain::ipv4, Type::datagram};

    auto creation_result = create_socket();

    BOOST_TEST(static_cast<bool>(creation_result));

    SocketFd socket_fd {move(*creation_result)};
    BOOST_TEST(socket_fd.domain() == create_socket.domain());
    BOOST_TEST(socket_fd.type() == create_socket.type_value());
    BOOST_TEST(socket_fd.protocol() == create_socket.protocol());
  }
  {
    CreateSocket create_socket {Domain::unix_, Type::stream};

    auto creation_result = create_socket();

    BOOST_TEST(static_cast<bool>(creation_result));

    SocketFd socket_fd {move(*creation_result)};
    BOOST_TEST(socket_fd.domain() == create_socket.domain());
    BOOST_TEST(socket_fd.type() == create_socket.type_value());
    BOOST_TEST(socket_fd.protocol() == create_socket.protocol());
  }
}

BOOST_AUTO_TEST_SUITE_END() // CreateSocket_tests
BOOST_AUTO_TEST_SUITE_END() // Sockets
BOOST_AUTO_TEST_SUITE_END() // IPC