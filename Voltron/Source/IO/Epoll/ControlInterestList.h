//------------------------------------------------------------------------------
/// \file ControlInterestList.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://man7.org/linux/man-pages/man2/epoll_ctl.2.html
/// \brief Add, modify, or remove entries in interest list of ::epoll instance.
//------------------------------------------------------------------------------
#ifndef IO_EPOLL_CONTROL_INTEREST_LIST
#define IO_EPOLL_CONTROL_INTEREST_LIST

#include "EpollFd.h"
#include "Utilities/ErrorHandling/ErrorHandling.h"

namespace IO
{
namespace Epoll
{

namespace Details
{

//------------------------------------------------------------------------------
/// \brief enum class for all control operations to be performed for a target
/// fd.
/// \ref http://man7.org/linux/man-pages/man2/epoll_ctl.2.html
//------------------------------------------------------------------------------
enum class ControlOperations : int
{
  add = EPOLL_CTL_ADD, // register target `fd` on `epoll` instance, `epfd`.
  modify = EPOLL_CTL_MOD, // change event `event` associated with `fd`
  remove = EPOLL_CTL_DEL // remove (degister) target fd from epoll instance.
};

} // namespace Details

//------------------------------------------------------------------------------
/// \brief enum class for all event types.
/// \details Events member of epoll_event is a bit mask composed by ORing
/// together 0 or more of the following available event types.
/// \ref http://man7.org/linux/man-pages/man2/epoll_ctl.2.html
//------------------------------------------------------------------------------
enum class EventTypes : uint32_t
{
  default_value = 0,
  read = EPOLLIN, // associated file available for read operations
  write = EPOLLOUT, // associated file available for write operations
  // stream socket peer closed connection, or shut down writing half of
  // connection (useful to detect peer shutdown when Edge triggered monitoring)
  stream_or_half_hangup = EPOLLRDHUP, 
  exceptional = EPOLLPRI, // exceptional condition on fd.
  error = EPOLLERR, // error condition happened on associated fd.
  hangup = EPOLLHUP, // hang up happened on associated fd; epoll_wait will wait
  edge_triggered = EPOLLET, // edge triggered behavior for associated fd
  one_shot = EPOLLONESHOT, // sets 1-shot behavior for associated fd.
  wakeup = EPOLLWAKEUP, // ensure system doesn't "suspend" while event pending
  exclusive = EPOLLEXCLUSIVE // sets exclusive wakeup mode for epoll fd.
};

class ControlInterestList
{
  public:

    using OptionalErrorNumber =
      Utilities::ErrorHandling::HandleReturnValuePassively::OptionalErrorNumber;

    ControlInterestList(const EpollFd& epoll_fd);

    ControlInterestList(const EpollFd& epoll_fd, const uint32_t events_type);

    ControlInterestList(const EpollFd& epoll_fd, const EventTypes event_type);

    void add_additional_event(const EventTypes event_type);

    // Accessors

    uint32_t events() const
    {
      return events_;
    }

    OptionalErrorNumber remove_from_interest_list(const int fd);

  private:

    // file descriptor referring to epoll instance.
    int epfd_;
    // Epoll events; can be zero or more of the available event types.
    uint32_t events_;
};

} // namespace Epoll
} // namespace IO

#endif // IO_EPOLL_CONTROL_INTEREST_LIST