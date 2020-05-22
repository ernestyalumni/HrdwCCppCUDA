//------------------------------------------------------------------------------
/// \file EpollFd_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
//------------------------------------------------------------------------------
#include "IO/Epoll/EpollFd.h"

#include <boost/test/unit_test.hpp>

using IO::Epoll::EpollFd;

BOOST_AUTO_TEST_SUITE(IO)
BOOST_AUTO_TEST_SUITE(Epoll)
BOOST_AUTO_TEST_SUITE(EpollFd_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
	EpollFd epoll_fd {};

	BOOST_TEST(epoll_fd.fd() >= 0);
}

BOOST_AUTO_TEST_SUITE_END() // EpollFd_tests
BOOST_AUTO_TEST_SUITE_END() // Epoll
BOOST_AUTO_TEST_SUITE_END() // IO