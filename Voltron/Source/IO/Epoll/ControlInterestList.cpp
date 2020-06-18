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

using Cpp::Utilities::TypeSupport::get_underlying_value;
using IO::Epoll::Details::ControlOperations;
using Utilities::ErrorHandling::HandleReturnValuePassively;

namespace IO
{
namespace Epoll
{

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
  const EventTypes event_type
  ):
  epfd_{epoll_fd.fd()},
  events_{get_underlying_value(event_type)}
{}

void ControlInterestList::add_additional_event(const EventTypes event_type)
{
  events_ |= get_underlying_value(event_type); 
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
      get_underlying_value(ControlOperations::remove),
      fd,
      nullptr)};

  return HandleReturnValuePassively()(return_result);  
}


} // namespace Epoll
} // namespace IO