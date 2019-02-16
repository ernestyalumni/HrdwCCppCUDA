//------------------------------------------------------------------------------
/// \file Events.h
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
#ifndef _IO_EPOLL_EVENT_H_
#define _IO_EPOLL_EVENT_H_

#include <ostream>
#include <sys/epoll.h>

namespace IO
{

namespace Epoll
{

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
  stream_or_hangup_half = EPOLLRDHUP, 
  exceptional = EPOLLPRI, // exceptional condition on fd.
  error = EPOLLERR, // error condition happened on associated fd.
  hangup = EPOLLHUP, // hang up happened on associated fd; epoll_wait will wait
  edge_triggered = EPOLLET, // edge triggered behavior for associated fd
  one_shot = EPOLLONESHOT, // sets 1-shot behavior for associated fd.
  wakeup = EPOLLWAKEUP, // ensure system doesn't "suspend" while event pending
  exclusive = EPOLLEXCLUSIVE // sets exclusive wakeup mode for epoll fd.
};

//------------------------------------------------------------------------------
/// \brief Derived class for ::epoll_event
/// \details struct epoll_event is defined as 
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
/// pp. 1357, Ch. 63 Kerrisk, The Linux Programming Interface (2010).
//------------------------------------------------------------------------------
struct EpollEvent : public ::epoll_event
{
  EpollEvent(const EventTypes& event_type, const int fd);

  EpollEvent(const uint32_t events, const int fd);

  explicit EpollEvent(const EventTypes& event_type);

  explicit EpollEvent(const uint32_t events);

  EpollEvent() = default;

  // Consider using &epoll_event for Epollevent epoll_event, instead.
  const ::epoll_event* to_epoll_event() const
  {
    return reinterpret_cast<const ::epoll_event*>(this);
  }

  ::epoll_event* to_epoll_event()
  {
    return reinterpret_cast<::epoll_event*>(this);
  }

  friend std::ostream& operator<<(
    std::ostream& os,
    const EpollEvent& epoll_event);
};

//------------------------------------------------------------------------------
/// \class Event
/// \brief ::epoll_event using encapsulation (composition)
//------------------------------------------------------------------------------
class Event
{
  public:

  Event(const EventTypes& event_type, const int fd);

  Event(const uint32_t events, const int fd);

  explicit Event(const EventTypes& event_type);

  explicit Event(const uint32_t events);

  Event() = default;

  //----------------------------------------------------------------------------
  /// \brief user-defined conversion
  /// \details Make a reference when user wants to pass this user-defined
  /// converted Event into a function that takes a pointer, e.g.
  ///
  /// &(::epoll_event{some_event})
  ///
  /// \ref https://en.cppreference.com/w/cpp/language/cast_operator
  //----------------------------------------------------------------------------
  operator ::epoll_event() const
  {
    return epoll_event_;
  }

  /// Accessors for Linux system call.

  const ::epoll_event* to_epoll_event_pointer() const
  {
    return &epoll_event_;
  }

  ::epoll_event* to_epoll_event_pointer()
  {
    return &epoll_event_;
  }

  const ::epoll_event* as_epoll_event_pointer() const
  {
    return reinterpret_cast<const ::epoll_event*>(this);
  }

  ::epoll_event* as_epoll_event_pointer()
  {
    return reinterpret_cast<::epoll_event*>(this);
  }

  friend std::ostream& operator<<(std::ostream& os, const Event& event);

  private:

    ::epoll_event epoll_event_;
};

} // namespace Epoll

} // namespace IO

#endif // _IO_EPOLL_EVENT_H_