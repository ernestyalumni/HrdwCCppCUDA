//------------------------------------------------------------------------------
/// \file Event_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Wrappers and examples classes for errno and error handling.
/// \ref https://linux.die.net/man/2/epoll_ctl
/// https://en.cppreference.com/w/cpp/error/errno_macros
/// \details Event describes object linked to fd.  struct epoll_event.
/// \copyright If you find this code useful, feel free to donate directly
/// (username ernestyalumni or email address above), going directly to:
///
/// paypal.me/ernestyalumni
///
/// which won't go through a 3rd. party like indiegogo, kickstarter, patreon.
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
///   g++ --std=c++17 -I ../../ Event.cpp Event_main.cpp -o Event_main
//------------------------------------------------------------------------------
#include "Event.h"

#include "Utilities/casts.h" // get_underlying

#include <iostream>

using IO::Epoll::EpollEvent;
using IO::Epoll::Event;
using IO::Epoll::EventTypes;
using Utilities::get_underlying_value;

int main()
{
  // EventTypes
  {
    std::cout << "\n EventTypes \n";
    std::cout << " EventTypes::default_value : " <<
      static_cast<int>(EventTypes::default_value) << '\n'; // 0 
    std::cout << " EventTypes::read : " <<
      static_cast<int>(EventTypes::read) << '\n'; // 1
    std::cout << " EventTypes::write : " <<  
      static_cast<int>(EventTypes::write) << '\n'; // 4 
    std::cout << " EventTypes::stream_or_hangup_half : " <<
      static_cast<int>(EventTypes::stream_or_hangup_half) << '\n'; // 8192
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

    const EpollEvent epoll_event_1 {EventTypes::read, 6};
    std::cout << " epoll_event_1 : " << epoll_event_1 << '\n';

    const EpollEvent epoll_event_2 {
      get_underlying_value<EventTypes>(EventTypes::read) ||
        get_underlying_value<EventTypes>(EventTypes::edge_triggered),
     7};

    std::cout << " epoll_event_2 : " << epoll_event_2 << '\n';
  }

 // EventConstructs
  {
    std::cout << "\n EventConstructs \n";

    const Event event {
      get_underlying_value<EventTypes>(EventTypes::default_value),
      5};
    std::cout << " event : " << event << '\n';

    const Event event_1 {EventTypes::read, 6};
    std::cout << " event_1 : " << event_1 << '\n';

    const Event event_2 {
      get_underlying_value<EventTypes>(EventTypes::read) ||
        get_underlying_value<EventTypes>(EventTypes::edge_triggered),
     7};

    std::cout << " event_2 : " << event_2 << '\n';
  }


}