//------------------------------------------------------------------------------
/// \file EpollFd_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
//------------------------------------------------------------------------------
#include "IO/Epoll/EpollFd.h"
#include "UnitTests/Tools/Contains.h"
#include "Utilities/ErrorHandling/ErrorNumber.h"

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <sys/resource.h>
#include <system_error> // std::system_error

using IO::Epoll::EpollFd;
using UnitTests::Tools::Contains;
using UnitTests::Tools::error_contains;
using Utilities::ErrorHandling::ErrorNumber;

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

// ::getrlimit, ::setrlimit
// cf. https://man7.org/linux/man-pages/man2/getrlimit.2.html
// int getrlimit(int resource, struct rlimit *rlim);
// int setrlimit(int resource, const struct rlimit* rlim);
//
// On success, these system calls return 0. On error, -1 returned and errno set.
//
// resource argument must be 1 of:
// RLIMIT_AS - This is max size of process's virtual memory (address space).
// Limit specified in bytes, rounded down to system page size.
// RLIMIT_CORE - max size of core file in bytes that process may dump.
// RLIMIT_CPU - this is limit, in seconds, on amount of CPU time that process
// can consume.
//
// RLIMIT_NOFILE - specifies a value 1 greater than max fd number that can be
// opened by this process. Attempts (::open, ::pipe, ::dup, etc.) to exceed this
// limit yield error EMFILE.
//
// RLIMIT_NPROC - This is limit on number of extant process (or, more precisely
// on Linux, threads) for real user ID of calling process.

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DoesNotConstructIf)
{
  // Each resource has associated soft and hard limit, as defined by rlimit
  // structure:
  // struct rlimit {
  //  rlim_t rlim_cur; // Soft limit
  //  rlim_t rlim_max; // Hard limit (ceiling for rlim_cur)
  // };
  // soft limit is value that kernel enforces for corresponding resource.
  // hard limit acts as ceiling for soft limit: an unprivileged process may set
  // only its soft limit to value in range from 0 up to hard limit, and
  // (irreversibly) lower its hard limit.
  // 
  // A privileged process (uner Linux: 1 with CAP_SYS_RESOURCE capability in
  // initial user namespace) may make arbitrary changes to either limit value.
  //
  ::rlimit resource_limits {0, 0};

  // Address space limit
  {
    int result_value{::getrlimit(RLIMIT_AS, &resource_limits)};

    BOOST_TEST_REQUIRE(result_value == 0);
    std::cout << "\n resource_limits.rlim_cur: " << resource_limits.rlim_cur <<
      "\n"; // 18446744073709551615 0xffff ffff ffff ffff 8 bytes
    std::cout << "\n resource_limits.rlim_max: " << resource_limits.rlim_max <<
      "\n";
  }
  // Core file.
  {
    resource_limits = {0, 0};
    BOOST_TEST_REQUIRE(resource_limits.rlim_cur == 0);
    const int result_value{::getrlimit(RLIMIT_CORE, &resource_limits)};

    BOOST_TEST_REQUIRE(result_value == 0);
    std::cout << "\n resource_limits.rlim_cur: " << resource_limits.rlim_cur <<
      "\n"; // 18446744073709551615 0xffff ffff ffff ffff 8 bytes
    std::cout << "\n resource_limits.rlim_max: " << resource_limits.rlim_max <<
      "\n";
  }
  // CPU limit (in time, seconds).
  {
    resource_limits = {0, 0};
    BOOST_TEST_REQUIRE(resource_limits.rlim_cur == 0);
    const int result_value{::getrlimit(RLIMIT_CPU, &resource_limits)};

    BOOST_TEST_REQUIRE(result_value == 0);
    std::cout << "\n resource_limits.rlim_cur: " << resource_limits.rlim_cur <<
      "\n"; // 18446744073709551615 0xffff ffff ffff ffff 8 bytes
    std::cout << "\n resource_limits.rlim_max: " << resource_limits.rlim_max <<
      "\n";
  }
  // Number of extant process or threads limit for process.
  {
    resource_limits = {0, 0};
    BOOST_TEST_REQUIRE(resource_limits.rlim_cur == 0);
    const int result_value{::getrlimit(RLIMIT_NPROC, &resource_limits)};

    BOOST_TEST_REQUIRE(result_value == 0);
    std::cout << "\n resource_limits.rlim_cur: " << resource_limits.rlim_cur <<
      "\n"; // 127812
    std::cout << "\n resource_limits.rlim_max: " << resource_limits.rlim_max <<
      "\n";
  }

  // Number of opened fds limit for process.
  {
    resource_limits = {0, 0};
    BOOST_TEST_REQUIRE(resource_limits.rlim_cur == 0);
    const int result_value{::getrlimit(RLIMIT_NOFILE, &resource_limits)};

    BOOST_TEST_REQUIRE(result_value == 0);
    std::cout << "\n resource_limits.rlim_cur: " << resource_limits.rlim_cur <<
      "\n"; // 1024
    std::cout << "\n resource_limits.rlim_max: " << resource_limits.rlim_max <<
      "\n"; // 524288
  }

  ::rlimit new_resource_limits {resource_limits};
  // This effectively sets number of fds to 0.
  new_resource_limits.rlim_cur = 1;

  BOOST_TEST_REQUIRE(::setrlimit(RLIMIT_NOFILE, &new_resource_limits) == 0);

  // BOOST_<level>_THROW
  // BOOST_WARN_THROW(expression, exception_type);
  // BOOST_CHECK_THROW(expression, exception_type);
  // cf. https://www.boost.org/doc/libs/1_64_0/libs/test/doc/html/boost_test/utf_reference/testing_tool_ref/assertion_boost_level_throw.html
  BOOST_CHECK_THROW(EpollFd{}, std::system_error);

  BOOST_CHECK_EXCEPTION(
    EpollFd{5},
    std::system_error,
    error_contains("Too many open files"));

  //std::cout << "\n error no : " << ErrorNumber{}.as_string() << "\n";

  // Remember to reset the limits for this resource.

  BOOST_TEST_REQUIRE(::setrlimit(RLIMIT_NOFILE, &resource_limits) == 0);
  BOOST_TEST_REQUIRE(::getrlimit(RLIMIT_NOFILE, &resource_limits) == 0);

  std::cout << "\n resource_limits.rlim_cur: " << resource_limits.rlim_cur <<
    "\n"; // 1024
  std::cout << "\n resource_limits.rlim_max: " << resource_limits.rlim_max <<
    "\n"; // 524288  
}

BOOST_AUTO_TEST_SUITE_END() // EpollFd_tests
BOOST_AUTO_TEST_SUITE_END() // Epoll
BOOST_AUTO_TEST_SUITE_END() // IO