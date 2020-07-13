//------------------------------------------------------------------------------
/// \file ControlInterestList.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://man7.org/linux/man-pages/man2/epoll_ctl.2.html
/// \brief Add, modify, or remove entries in interest list of ::epoll instance.
//------------------------------------------------------------------------------
#include "ControlInterestList.h"

#include "Cpp/Utilities/TypeSupport/UnderlyingTypes.h"
#include "Utilities/ErrorHandling/ErrorHandling.h"

#include <ostream>
#include <sys/epoll.h>
#include <utility> // std::move, std::pair

using Cpp::Utilities::TypeSupport::get_underlying_value;
using IO::Epoll::Details::ControlOperations;
using Utilities::ErrorHandling::HandleReturnValuePassively;

namespace IO
{
namespace Epoll
{

namespace Details
{

EpollEvent::EpollEvent(const EventTypes event_type, const int fd):
  ::epoll_event{get_underlying_value(event_type)}
{
  this->data.fd = fd;
}

EpollEvent::EpollEvent(const uint32_t events, const int fd):
    ::epoll_event{events}
{
  this->data.fd = fd;
}

EpollEvent::EpollEvent(const EventTypes event_type):
  ::epoll_event{get_underlying_value(event_type)}
{}

EpollEvent::EpollEvent(const uint32_t events):
  ::epoll_event{events}
{}

EpollEvent::EpollEvent() = default;

std::ostream& operator<<(
  std::ostream& os,
  const EpollEvent& epoll_event)
{
  os << epoll_event.events << ' ' << epoll_event.data.fd << ' ' <<
    epoll_event.data.u32 << ' ' << epoll_event.data.u64 << '\n';

  return os;
}

} // namespace Details

ControlInterestList::ControlInterestList(const EpollFd& epoll_fd):
  epfd_{epoll_fd.fd()},
  events_{}
{}

ControlInterestList::ControlInterestList(
  const EpollFd& epoll_fd,
  const uint32_t events_type
  ):
  epfd_{epoll_fd.fd()},
  events_{events_type}
{}

ControlInterestList::ControlInterestList(
  const EpollFd& epoll_fd,
  const Details::EventTypes event_type
  ):
  epfd_{epoll_fd.fd()},
  events_{get_underlying_value(event_type)}
{}

void ControlInterestList::add_additional_event(
  const Details::EventTypes event_type)
{
  events_ |= get_underlying_value(event_type); 
}

std::pair<ControlInterestList::OptionalErrorNumber, Details::EpollEvent>
  ControlInterestList::add_to_interest_list(const int fd)
{
  Details::EpollEvent epoll_event_for_adding {events_, fd};

  const int return_result {
    ::epoll_ctl(
      epfd_,
      get_underlying_value(Details::ControlOperations::add),
      fd,
      &(epoll_event_for_adding.as_epoll_event()))};

  return std::make_pair<OptionalErrorNumber, Details::EpollEvent>(
    HandleReturnValuePassively()(return_result),
    std::move(epoll_event_for_adding));
}

std::pair<ControlInterestList::OptionalErrorNumber, Details::EpollEvent>
  ControlInterestList::modify_settings(const int fd)
{
  Details::EpollEvent epoll_event_for_modifying {events_, fd};

  const int return_result {
    ::epoll_ctl(
      epfd_,
      get_underlying_value(Details::ControlOperations::modify),
      fd,
      &(epoll_event_for_modifying.as_epoll_event()))};

  return std::make_pair<OptionalErrorNumber, Details::EpollEvent>(
    HandleReturnValuePassively()(return_result),
    std::move(epoll_event_for_modifying));
}

ControlInterestList::OptionalErrorNumber
  ControlInterestList::remove_from_interest_list(const int fd)
{
  // https://man7.org/linux/man-pages/man2/epoll_ctl.2.html
  // BUGS: Since Linux 2.6.9, event can be specified as NULL when using
  // EPOLL_CTL_DEL.
  const int return_result {
    ::epoll_ctl(
      epfd_,
      get_underlying_value(Details::ControlOperations::remove),
      fd,
      nullptr)};

  return HandleReturnValuePassively()(return_result);  
}

} // namespace Epoll
} // namespace IO