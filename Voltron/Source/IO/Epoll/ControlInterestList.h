//------------------------------------------------------------------------------
/// \file ControlInterestList.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://man7.org/linux/man-pages/man2/epoll_ctl.2.html
/// \brief Add, modify, or remove entries in interest list of ::epoll instance.
//------------------------------------------------------------------------------
#ifndef IO_EPOLL_CONTROL_INTEREST_LIST
#define IO_EPOLL_CONTROL_INTEREST_LIST

#include "Cpp/Utilities/TypeSupport/UnderlyingTypes.h"
#include "EpollFd.h"
#include "Utilities/ErrorHandling/ErrorHandling.h"

#include <ostream>
#include <sys/epoll.h>
#include <utility> // std::pair

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
  write = EPOLLOUT, // associated file available for write operations.
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

//------------------------------------------------------------------------------
/// \struct EpollEvent
/// \brief Derived class for ::epoll_event
/// \details Describes the object linked to file descriptor fd.
/// struct epoll_event is defined as 
/// 
/// typedef union epoll_data {
///   void *ptr; // pointer to user-defined data
///   int fd;
///   uint32_t u32;
///   uint64_t u64;
/// } epoll_data_t
/// 
/// struct epoll_event
/// {
///   uint32_t events;    // Epoll events, bit mask specifying set of events
///   epoll_data_t data;  // User data variable
/// }
/// 
/// \ref http://man7.org/linux/man-pages/man2/epoll_ctl.2.html
/// https://stackoverflow.com/questions/11916707/how-to-use-epoll-event-data-ptr
/// pp. 1357, Ch. 63 Kerrisk, The Linux Programming Interface (2010).
//------------------------------------------------------------------------------
struct EpollEvent : public ::epoll_event
{
  EpollEvent(const EventTypes event_type, const int fd);

  EpollEvent(const uint32_t events, const int fd);

  explicit EpollEvent(const EventTypes event_type);

  explicit EpollEvent(const uint32_t events);

  EpollEvent();

  // Consider using &epoll_event for Epollevent epoll_event, instead.
  const ::epoll_event* to_epoll_event() const
  {
    return reinterpret_cast<const ::epoll_event*>(this);
  }

  ::epoll_event* to_epoll_event()
  {
    return reinterpret_cast<::epoll_event*>(this);
  }

  const ::epoll_event as_epoll_event() const
  {
    return ::epoll_event{reinterpret_cast<const ::epoll_event&>(*this)};
  }

  ::epoll_event& as_epoll_event()
  {
    return reinterpret_cast<::epoll_event&>(*this);
  }

  friend std::ostream& operator<<(
    std::ostream& os,
    const EpollEvent& epoll_event);
};

std::ostream& operator<<(std::ostream& os, const EpollEvent& epoll_event);

} // namespace Details

//------------------------------------------------------------------------------
/// \class ControlInterestList
/// \ref https://man7.org/linux/man-pages/man2/epoll_ctl.2.html
/// \details #include <sys/epoll.h>
/// int epoll_ctl(int epfd, int op, int fd, struct epoll_event* event)
///
/// Valid values for op argument are:
/// EPOLL_CTL_ADD - add to interest list of epfd, settings specified in event
/// EPOLL_CTL_MOD - change settings associated with fd in interest list to
/// new settings specified in event.
/// EPOLL_CTL_DEL - remove (deregister) target fd from interest list. event
/// argument ignored and can be NULL
//------------------------------------------------------------------------------

class ControlInterestList
{
  public:

    using OptionalErrorNumber =
      Utilities::ErrorHandling::HandleReturnValuePassively::OptionalErrorNumber;

    ControlInterestList(const EpollFd& epoll_fd);

    ControlInterestList(const EpollFd& epoll_fd, const uint32_t events_type);

    ControlInterestList(
      const EpollFd& epoll_fd,
      const Details::EventTypes event_type);

    void add_additional_event(const Details::EventTypes event_type);

    // Accessors

    uint32_t events() const
    {
      return events_;
    }

    std::pair<OptionalErrorNumber, Details::EpollEvent> add_to_interest_list(
      const int fd);

    std::pair<OptionalErrorNumber, Details::EpollEvent> modify_settings(
      const int fd);

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