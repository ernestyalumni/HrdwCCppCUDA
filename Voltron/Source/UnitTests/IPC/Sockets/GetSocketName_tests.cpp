//------------------------------------------------------------------------------
/// \file GetSocketName_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
//------------------------------------------------------------------------------
#include "IPC/Sockets/GetSocketName.h"

#include "IPC/Sockets/Bind.h"
#include "IPC/Sockets/InternetAddress.h"
#include "IPC/Sockets/ParameterFamilies.h"
#include "IPC/Sockets/Socket.h"

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <tuple>

using IPC::Sockets::Bind;
using IPC::Sockets::Domain;
using IPC::Sockets::GetSocketName;
using IPC::Sockets::InternetSocketAddress;
using IPC::Sockets::Socket;
using IPC::Sockets::Type;

BOOST_AUTO_TEST_SUITE(IPC)
BOOST_AUTO_TEST_SUITE(Sockets)
BOOST_AUTO_TEST_SUITE(GetSocketName_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(GetSocketNameGetsInternetAddress)
{
  {
    Socket socket {Domain::ipv4, Type::datagram};
    BOOST_TEST_REQUIRE(socket.domain() == AF_INET);
    BOOST_TEST_REQUIRE(socket.type() == SOCK_DGRAM);

    const InternetSocketAddress internet_socket_address {0};

    Bind bind_f {internet_socket_address};
    const auto bind_result = bind_f(socket);

    BOOST_TEST_REQUIRE(!static_cast<bool>(bind_result));

    GetSocketName get_socket_name {};

    const auto result = get_socket_name(socket);

    if (result)
    {
      std::cout << "\n GetSocketName Failure: " << (*result).as_string() <<
        " error number: " << (*result).error_number() << "\n";
    }

    // TODO: determine how this fails.
    /*
    BOOST_TEST_REQUIRE(!static_cast<bool>(result));

    BOOST_TEST(static_cast<bool>(get_socket_name.socket_address()));

    const InternetSocketAddress returned_address {
      std::get<0>(*(get_socket_name.socket_address()))};

    BOOST_TEST(returned_address.sin_port > -1);
    BOOST_TEST(::ntohs(returned_address.sin_port) > -1);

    std::cout << "Bind complete. Port number = " << returned_address.sin_port <<
      " : " << ::ntohs(returned_address.sin_port) << "\n";
    */
  }
}

BOOST_AUTO_TEST_SUITE_END() // GetSocketName_tests
BOOST_AUTO_TEST_SUITE_END() // Sockets
BOOST_AUTO_TEST_SUITE_END() // IPC