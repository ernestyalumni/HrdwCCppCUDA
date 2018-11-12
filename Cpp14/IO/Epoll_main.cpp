//------------------------------------------------------------------------------
/// \file Epoll_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  epoll as RAII main driver file.
/// \ref http://man7.org/linux/man-pages/man2/epoll_create.2.html     
/// \details Using RAII for epoll instance. 
/// \copyright If you find this code useful, feel free to donate directly and easily at 
/// this direct PayPal link: 
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
///  g++ -std=c++14 Epoll_main.cpp -o Epoll_main
//------------------------------------------------------------------------------
#include "Epoll.h"

#include "../Utilities/casts.h" // get_underlying

#include <iostream>
#include <sys/epoll.h>

using IO::ControlOperations;
using IO::Epoll;
using IO::EpollEvent;
using IO::EpollFlags;
using IO::EventTypes;
using Utilities::get_underlying_value;

template <EpollFlags EpollFlag = EpollFlags::default_value>
class TestableEpoll : public Epoll<EpollFlag>
{
  public:

    // Inherit default ctor
    using Epoll<EpollFlag>::Epoll;

    using Epoll<EpollFlag>::add_fd;
    using Epoll<EpollFlag>::fd;
};

int main()
{
  // EpollFlags
  {
    std::cout << "\n EpollFlags \n";
    std::cout << " EpollFlags::default_value : " <<
      static_cast<int>(EpollFlags::default_value) << '\n'; // 0
    std::cout << " EpollFlags::close_on_execute : " <<
      static_cast<int>(EpollFlags::close_on_execute) << '\n'; // 524288
  }

  // ControlOperations
  {
    std::cout << "\n ControlOperations \n";
    std::cout << " ControlOperations::add : " <<
      static_cast<int>(ControlOperations::add) << '\n'; // 1
    std::cout << " ControlOperations::modify : " <<
      static_cast<int>(ControlOperations::modify) << '\n'; // 3
    std::cout << " ControlOperations::remove : " <<
      static_cast<int>(ControlOperations::remove) << '\n'; // 2
  }

  // EventTypes
  {
    std::cout << "\n EventTypes \n";
    std::cout << " EventTypes::default_value : " <<
      static_cast<int>(EventTypes::default_value) << '\n'; // 0 
    std::cout << " EventTypes::read : " <<
      static_cast<int>(EventTypes::read) << '\n'; // 1
    std::cout << " EventTypes::write : " <<  
      static_cast<int>(EventTypes::write) << '\n'; // 4 
    std::cout << " EventTypes::stream_or_half_hangup : " <<
      static_cast<int>(EventTypes::stream_or_half_hangup) << '\n'; // 8192
    std::cout << " EventTypes::exceptional : " <<
      static_cast<int>(EventTypes::exceptional) << '\n'; // 2 
    std::cout << " EventTypes::error : " <<
      static_cast<int>(EventTypes::error) << '\n'; // 8
    std::cout << " EventTypes::hangup : " <<
      static_cast<int>(EventTypes::hangup) << '\n'; // 16
    std::cout << " EventTypes::edge_triggered : " <<
      static_cast<int>(EventTypes::edge_triggered) << '\n'; // -2147483648
    std::cout << " EventTypes::one_shot : " <<
      static_cast<int>(EventTypes::one_shot) << '\n'; // 1073741824
    std::cout << " EventTypes::wakeup : " <<
      static_cast<int>(EventTypes::wakeup) << '\n'; // 536870912
    std::cout << " EventTypes::exclusive : " <<
      static_cast<int>(EventTypes::exclusive) << '\n'; // 268435456
  }

  // EpollEventConstructs
  {
    std::cout << "\n EpollEventConstructs \n";

    const EpollEvent epoll_event {
      get_underlying_value<EventTypes>(EventTypes::default_value),
      5};
    std::cout << " epoll_event : " << epoll_event << '\n';

//    const EpollEvent<get_underlying_value<EventTypes>(EventTypes::read)>
// WORKS
//      static_cast<std::underlying_type_t<EventTypes>>(EventTypes::read)> 
    const EpollEvent epoll_event_1 {EventTypes::read, 5};
    std::cout << " epoll_event_1 : " << epoll_event_1 << '\n';

    const EpollEvent epoll_event_2 {
      get_underlying_value<EventTypes>(EventTypes::read) ||
        get_underlying_value<EventTypes>(EventTypes::edge_triggered),
      5};

    std::cout << " epoll_event_2 : " << epoll_event_2 << '\n';
  }

  std::cout << get_underlying_value<EventTypes>(EventTypes::read) << '\n';
  std::cout << get_underlying_value<EventTypes>(EventTypes::edge_triggered) <<
    '\n';
  std::cout << (get_underlying_value<EventTypes>(EventTypes::read)
    || get_underlying_value<EventTypes>(EventTypes::edge_triggered)) << '\n';

  std::cout << (get_underlying_value<EventTypes>(EventTypes::write)
    || get_underlying_value<EventTypes>(EventTypes::exclusive)) << '\n';

  std::cout << " EPOLLIN : " << EPOLLIN << '\n';
  std::cout << " EPOLLOUT : " << EPOLLOUT << '\n';
  std::cout << " EPOLLET : " << EPOLLET << '\n';
  std::cout << " EPOLLEXCLUSIVE : " << EPOLLEXCLUSIVE << '\n';

  std::cout << (EPOLLIN || EPOLLET) << '\n';
  std::cout << (EPOLLOUT || EPOLLEXCLUSIVE) << '\n';


  // EpollDefaultConstructs
  {
    std::cout << " \n EpollDefaultConstructs \n";
    const Epoll<> epoll;

    const TestableEpoll<> test_epoll;

    std::cout << " test_epoll.fd() : " << test_epoll.fd() << '\n';
  }

  // AddFdAddsFdIntoEpollSet
  {
    std::cout << "\n AddFdAddsFdIntoEpollSet \n";
  }

}
