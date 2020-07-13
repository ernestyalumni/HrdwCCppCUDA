//------------------------------------------------------------------------------
/// \file ControlInterestList_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
//------------------------------------------------------------------------------
#include "IO/Epoll/ControlInterestList.h"

#include "IO/Epoll/EpollFd.h"
#include "IPC/Sockets/ParameterFamilies.h"
#include "IPC/Sockets/Socket.h"

#include <boost/test/unit_test.hpp>
#include <sys/epoll.h>

using IO::Epoll::ControlInterestList;
using IO::Epoll::Details::EventTypes;
using IO::Epoll::EpollFd;
using IPC::Sockets::Socket;
using IPC::Sockets::Domains;
using IPC::Sockets::Types;

BOOST_AUTO_TEST_SUITE(IO)
BOOST_AUTO_TEST_SUITE(Epoll)
BOOST_AUTO_TEST_SUITE(ControlInterestList_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithEpollFd)
{
  const EpollFd epoll_fd {};

  ControlInterestList control_interface {epoll_fd};

  BOOST_TEST(control_interface.events() == 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithEpollFdAndEventTypes)
{
  const EpollFd epoll_fd {};

  ControlInterestList control_interface {epoll_fd, EventTypes::read};

  BOOST_TEST(control_interface.events() == EPOLLIN);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AddsToInterestList)
{
  EpollFd epoll_fd {};
  epoll_fd.fd();

  const Socket socket1 {Domains::unix_, Types::stream};
  const Socket socket2 {Domains::unix_, Types::stream};

  ControlInterestList control_interface {epoll_fd, EventTypes::read};

  control_interface.add_to_interest_list(socket1.fd());
  control_interface.add_to_interest_list(socket2.fd());

  control_interface.remove_from_interest_list(socket2.fd());
  control_interface.remove_from_interest_list(socket1.fd());

  BOOST_TEST(true);
}

BOOST_AUTO_TEST_SUITE_END() // ControlInterestList_tests
BOOST_AUTO_TEST_SUITE_END() // Epoll
BOOST_AUTO_TEST_SUITE_END() // IO