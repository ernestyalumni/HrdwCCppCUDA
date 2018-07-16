//------------------------------------------------------------------------------
/// \file epoll_example.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Example of the use of the Linux epoll API.
/// \ref https://suchprogramming.com/epoll-in-3-easy-steps/
/// http://man7.org/tlpi/code/online/dist/altio/epoll_input.c.html
/// \details This program monitors file descriptors for input events. 
/// \copyright If you find this code useful, feel free to donate directly and
/// easily at this direct PayPal link: 
///
/// https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
/// 
/// which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
/// Otherwise, I receive emails and messages on how all my (free) material on
/// physics, math, and engineering have helped students with their studies, and
/// I know what it's like to not have money as a student, but love physics (or 
/// math, sciences, etc.), so I am committed to keeping all my material 
/// open-source and free, whether or not sufficiently crowdfunded, under the 
/// open-source MIT license: feel free to copy, edit, paste, make your own
/// versions, share, use as you wish.    
/// Peace out, never give up! -EY
//------------------------------------------------------------------------------
/// COMPILATION TIPS:
///  g++ -std=c++14 -Wall -Werror epoll_example.cpp -o epoll_example
//------------------------------------------------------------------------------
#include "Epoll/Epoll.h"

#include <array>
#include <cstring> // std::strncmp
#include <iostream>
#include <string>
#include <sys/epoll.h> // for epoll_create1(), epoll_ctl(), struct epoll_event
#include <unistd.h> // for close(), read()
#include <utility> // std::pair

using IO::ControlOperation;
using IO::Epoll;
using IO::EpollEvent;
using IO::EventTypes;
using IO::SingleEpoll;

template <int Flags = 0, unsigned int MaxEvents = IO::MAX_EVENTS>
class TestableEpoll : public Epoll<Flags, MaxEvents>
{
  public:
    using Epoll<Flags, MaxEvents>::Epoll;
    using Epoll<Flags, MaxEvents>::epoll_fd;
    using Epoll<Flags, MaxEvents>::close_all_fds;
    using Epoll<Flags, MaxEvents>::events_fd;
};

template <int Flags>
class TestableSingleEpoll : public SingleEpoll<Flags>
{
  public:
    using SingleEpoll<Flags>::SingleEpoll;
    using SingleEpoll<Flags>::fd;
};

int main()
{
  constexpr unsigned int MAX_EVENTS {5};

  int epoll_fd {::epoll_create1(0)};

  std::cout << "\n epoll_fd : " << epoll_fd << '\n';

  // SingleEpollConstructs
  {
    TestableSingleEpoll<0> test_single_epoll{};
    std::cout << "\n test_epoll.fd() : " << test_single_epoll.fd() << '\n';
  }

  // PrintOutEPOLL_CLOEXEC
  std::cout << "\n EPOLL_CLOEXEC : " << EPOLL_CLOEXEC << '\n';

  // ::epoll_eventConstructs
  {
    std::cout << "\n ::epoll_eventConstructs \n";
    ::epoll_event ev;
    ev.data.fd = 2;
    std::cout << " ev.data.fd : " << ev.data.fd << '\n';
    ev.events = EPOLLIN;
    std::cout << " ev.events : " << ev.events << " EPOLLIN : " << EPOLLIN << '\n';

//    ::epoll_event epoll_event {EPOLLIN, {nullptr, 3}};
//    ::epoll_data_t epoll_data {nullptr, .fd = 3}; // Error, too many initializers
    ::epoll_data_t epoll_data {.fd = 3};
    std::cout << " epoll_data.fd : " << epoll_data.fd << '\n';
  }

  // EpollEventConstructs
  {
//    EpollEvent<EPOLLIN> epoll_event;
    EpollEvent epoll_event {EPOLLIN};
    epoll_event.data.fd = 4;
    epoll_event.events = EPOLLIN;
  }

  // EpollEventAsParameterFor::epoll_ctl
  {
    std::cout << "\n EpollEventAsParameterFor::epoll_ctl \n";

    // Create epoll instance
    TestableSingleEpoll<0> test_single_epoll{};
    std::cout << "\n test_epoll.fd() : " << test_single_epoll.fd() << '\n';

    EpollEvent epoll_event {EPOLLIN};
    epoll_event.data.fd = 0;

    int result_epoll_ctl {
      ::epoll_ctl(
        test_single_epoll.fd(),
        EPOLL_CTL_ADD,
        0,
//        epoll_event.to_epoll_event())};
        &epoll_event)};

    std::cout << " result_epoll_ctl : " << result_epoll_ctl << '\n';
  }

  // EpollConstructs
  {
    std::cout << " \n EpollConstructs \n";
    TestableEpoll<0, MAX_EVENTS> testable_epoll {};
    std::cout << "\n testable_epoll.epoll_fd() : " <<
      testable_epoll.epoll_fd() << '\n';
  }

  // EpollAddsFdsWithControlFd
  {
    std::cout << " \n EpollAddsFdsWithControlFd \n";

    TestableEpoll<0, MAX_EVENTS> testable_epoll {};
    std::cout << "\n testable_epoll.epoll_fd() : " <<
      testable_epoll.epoll_fd() << '\n';

    EpollEvent epoll_event {EventTypes::read, 0};

    testable_epoll.control_fd<static_cast<int>(ControlOperation::add)>(
      0, epoll_event);
  }
#if 0
  // EpollPollsWithLevelTriggeringNotification
  // \ref https://suchprogramming.com/epoll-in-3-easy-steps/
  {
    std::cout << "\nEpollPollsWithLevelTriggeringNotification::epoll_wait\n";

//    constexpr unsigned int MAX_EVENTS {5};
    constexpr unsigned int READ_SIZE {10};

    int running {1}, event_count;
    size_t bytes_read;
    std::array<char, READ_SIZE + 1> read_buffer;

    std::array<EpollEvent, MAX_EVENTS> events;

    TestableEpoll<0, MAX_EVENTS> test_epoll {};
    test_epoll.add_fd_to_poll<static_cast<uint32_t>(EventTypes::read)>(0);

    while(running)
    {
      std::cout << "\n Polling for input ... \n";
//      event_count 
      event_count = ::epoll_wait(
        test_epoll.epoll_fd(),
        events.data(),
        MAX_EVENTS, 30000);

      std::cout << event_count << " ready events \n";

      for (int i {0}; i < event_count; i++)
      {
        std::cout << " Reading file descriptor " << events[i].data.fd <<
          " -- ";

        bytes_read = ::read(events[i].data.fd, read_buffer.data(), READ_SIZE);
        std::cout << bytes_read << " bytes read.\n";
        read_buffer[bytes_read] = '\0';
        std::cout << "Read " << std::string(read_buffer.data()) << '\n';

        if (!std::strncmp(read_buffer.data(), "stop\n", 5))
        {
          running = 0;
        }
      }
    }

    test_epoll.close_all_fds();
  }
#endif 
  // ReadyAndWaitWaitsForEvents
  // \ref https://suchprogramming.com/epoll-in-3-easy-steps/
  {
    std::cout << "\nReadyAndWaitWaitsForEvents\n";

    constexpr unsigned int READ_SIZE {10};

    int running {1}, event_count;
    size_t bytes_read;
    std::array<char, READ_SIZE + 1> read_buffer;

    TestableEpoll<0, MAX_EVENTS> test_epoll {};
    test_epoll.add_fd_to_poll<static_cast<uint32_t>(EventTypes::read)>(0);
    test_epoll.add_fd_to_poll<static_cast<uint32_t>(EventTypes::read)>(1);

    while(running)
    {
      std::cout << "\n Polling for input ... \n";
  
      event_count = test_epoll.ready_and_wait();    

      std::cout << event_count << " ready events (event_count) \n";

      for (int i {0}; i < event_count; i++)
      {
        std::cout << " i : " << i << '\n';
        std::cout << " Reading file descriptor " << test_epoll.events_fd(i) <<
          " -- ";

        bytes_read = ::read(
          test_epoll.events_fd(i),
          read_buffer.data(),
          READ_SIZE);
        std::cout << bytes_read << " bytes read.\n";
        read_buffer[bytes_read] = '\0';
        std::cout << "Read " << std::string(read_buffer.data()) << '\n';

        if (!std::strncmp(read_buffer.data(), "stop\n", 5))
        {
          running = 0;
        }
      } // END of for loop, for (int i {0}; i < event_count; i++)
    } // END of while loop, while(running)
    test_epoll.close_all_fds();
  }

  ::close(epoll_fd);
}

