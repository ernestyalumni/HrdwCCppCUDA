//------------------------------------------------------------------------------
/// \file Event.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Wrappers and examples classes for errno and error handling.
/// \ref https://linux.die.net/man/2/epoll_ctl
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

#include "Utilities/casts.h" // get_underlying_value

#include <ostream>

namespace IO
{

namespace Epoll
{

EpollEvent::EpollEvent(const EventTypes& event_type, const int fd):
  ::epoll_event{Utilities::get_underlying_value<EventTypes>(event_type)}
{
  this->data.fd = fd;
}

EpollEvent::EpollEvent(const uint32_t events, const int fd):
  ::epoll_event{events}
{
  this->data.fd = fd;
}

EpollEvent::EpollEvent(const EventTypes& event_type):
  ::epoll_event{Utilities::get_underlying_value<EventTypes>(event_type)}
{}

EpollEvent::EpollEvent(const uint32_t events):
  ::epoll_event{events}
{}

std::ostream& operator<<(std::ostream& os, const EpollEvent& epoll_event)
{
  os << epoll_event.events << ' ' << epoll_event.data.fd << ' ' <<
    epoll_event.data.u32 << ' ' << epoll_event.data.u64 << '\n';

  return os;
}

Event::Event(const EventTypes& event_type, const int fd):
  epoll_event_{Utilities::get_underlying_value<EventTypes>(event_type)}
{
  epoll_event_.data.fd = fd;
}

Event::Event(const uint32_t events, const int fd):
  epoll_event_{events}
{
  epoll_event_.data.fd = fd;
}

Event::Event(const EventTypes& event_type):
  epoll_event_{Utilities::get_underlying_value<EventTypes>(event_type)}
{}

Event::Event(const uint32_t events):
  epoll_event_{events}
{}

std::ostream& operator<<(std::ostream& os, const Event& event)
{
  os << event.epoll_event_.events << ' ' << event.epoll_event_.data.fd << ' ' <<
    event.epoll_event_.data.u32 << ' ' << event.epoll_event_.data.u64 << '\n';

  return os;
}


} // namespace Epoll

} // namespace IO