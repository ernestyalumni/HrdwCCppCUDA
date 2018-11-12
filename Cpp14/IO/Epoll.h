//------------------------------------------------------------------------------
/// \file Epoll.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  epoll as RAII (Resource Acquisition Is Initialization.
/// \ref http://man7.org/linux/man-pages/man2/epoll_create.2.html    
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
///
/// \details
//------------------------------------------------------------------------------
/// COMPILATION TIPS:
///  g++ -std=c++14 Epoll_main.cpp -o Epoll_main
//------------------------------------------------------------------------------
#ifndef _IO_EPOLL_H_
#define _IO_EPOLL_H_

#include "../Utilities/CheckReturn.h" // CheckReturn, check_close
#include "../Utilities/casts.h" // get_underlying_value

#include <algorithm> // std::find_if
#include <iostream>
#include <stdexcept> // std::out_of_range
#include <string> // std::to_string
#include <sys/epoll.h>
#include <type_traits> 
#include <unistd.h> // ::close
#include <utility> // std::pair, std::get
#include <vector>

namespace IO
{

//------------------------------------------------------------------------------
/// \brief enum class for all flags for the creation of a new epoll instance.
//------------------------------------------------------------------------------
enum class EpollFlags : int
{
  default_value = 0,
  close_on_execute = EPOLL_CLOEXEC
};

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


//------------------------------------------------------------------------------
/// \brief Derived class for ::epoll_event
///
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
  EpollEvent(const EventTypes event_type, const int fd):
    ::epoll_event{
      Utilities::get_underlying_value<EventTypes>(event_type)}
  {
    this->data.fd = fd;
  }

  EpollEvent(const uint32_t events, const int fd):
    ::epoll_event{events}
  {
    this->data.fd = fd;
  }

  explicit EpollEvent(const EventTypes event_type):
    ::epoll_event{
      Utilities::get_underlying_value<EventTypes>(event_type)}
  {}

  explicit EpollEvent(const uint32_t events):
    ::epoll_event{events}
  {}

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

std::ostream& operator<<(std::ostream& os, const EpollEvent& epoll_event)
{
  os << epoll_event.events << ' ' << epoll_event.data.fd << ' ' <<
    epoll_event.data.u32 << ' ' << epoll_event.data.u64 << '\n';

  return os;
}

template <EpollFlags EpollFlag = EpollFlags::default_value>
class Epoll
{
  public:

    using CheckClose = Utilities::CheckClose;
    using CheckReturn = Utilities::CheckReturn;

    Epoll();

    // Not copyable, not movable
    Epoll(const Epoll&) = delete; // copy ctor
    Epoll& operator=(const Epoll&) = delete; // copy assignment

    Epoll(Epoll&&);   // move ctor
    Epoll& operator=(Epoll&&); // move assignment

    ~Epoll()
    {
      CheckClose()(::close(fd_));
    }

    const std::vector<std::pair<int, EpollEvent>>::iterator find(const int fd)
      const
    {
      const auto find_result = std::find_if(
        set_of_fds_.begin(),
        set_of_fds_.end(),
        [fd](const std::pair<int, EpollEvent>& key_value) -> bool
        {
          return key_value.first == fd;
        });

      if (find_result != set_of_fds_.end())
      {
        return find_result;
      }
      else
      {
        throw std::out_of_range {"Could not find fd " + std::to_string(fd)};
      }
    }


    std::vector<std::pair<int, EpollEvent>>::iterator find(const int fd)
    {
      auto find_result = std::find_if(
        set_of_fds_.begin(),
        set_of_fds_.end(),
        [fd](const std::pair<int, EpollEvent>& key_value) -> bool
        {
          return key_value.first == fd;
        });

      if (find_result != set_of_fds_.end())
      {
        return find_result;
      }
      else
      {
        throw std::out_of_range {"Could not find fd " + std::to_string(fd)};
      }
    }

    //--------------------------------------------------------------------------
    /// \brief Accessor for diagnostic purposes, after using a system call.
    //--------------------------------------------------------------------------    
    EpollEvent last_epoll_event() const
    {
      return last_epoll_event_;
    }


  protected:

    void add_fd(const int fd, const EpollEvent& epoll_event)
    {
      control_operation<ControlOperations::add>(fd, epoll_event.to_epoll_event());
      set_of_fds_.emplace_back(std::pair<int, EpollEvent>(fd, epoll_event));
    }

    void remove_fd(const int fd)
    {
      auto key_value = this->find(fd);

      set_of_fds_.erase(key_value, key_value + 1);

      control_operation<ControlOperations::remove>(fd, nullptr);
    }

    int fd() const
    {
      return fd_;
    }

  private:

    class CheckWait;

    //--------------------------------------------------------------------------
    /// \brief Thin wrapper function for `::epoll_create1()`, that creates a 
    /// new `epoll()` instance. Checks system call for success or error.
    //--------------------------------------------------------------------------    
    int create_epoll_instance();

    //--------------------------------------------------------------------------
    /// \brief Thin wrapper function for `::epoll_ctl()`, that performs control 
    /// operations on epoll instance and operation op for target fd.
    /// \details Checks system call for success or error.
    //--------------------------------------------------------------------------    
    template <ControlOperations ControlOperation>
    int control_operation(const int fd, const EpollEvent& epoll_event);

    //--------------------------------------------------------------------------
    /// \brief Thin wrapper function for `::epoll_wait` that waits for an I/O
    /// event on an epoll fd.
    /// \param MaxEvents Maximum number of events returned by `epoll_wait()`.
    /// MaxEvents must be greater than 0.
    //--------------------------------------------------------------------------    
    template <
      int MaxEvents,
      int Timeout,
      typename = std::enable_if_t<(MaxEvents > 0) && (Timeout > -2)>
      >
    int wait(const EpollEvent& events);

    int fd_;

    std::vector<std::pair<int, EpollEvent>> set_of_fds_;

    // \brief For diagnostic purposes
    EpollEvent last_epoll_event_;
};

template <EpollFlags EpollFlag>
Epoll<EpollFlag>::Epoll():
  fd_{create_epoll_instance()}
{}

template <EpollFlags EpollFlag>
class Epoll<EpollFlag>::CheckWait : public CheckReturn
{
  public:

    CheckWait() = default;

    int operator()(int result)
    {
      number_of_ready_fds_ =
        this->operator()(result, "Failed to call wait (::epoll_wait())");
      return number_of_ready_fds_;
    }

    int number_of_ready_fds() const
    {
      return number_of_ready_fds_;
    }

  private:

    using Utilities::CheckReturn::operator();

    int number_of_ready_fds_; 
};


template <EpollFlags EpollFlag>
int Epoll<EpollFlag>::create_epoll_instance()
{
  return CheckReturn()(
    ::epoll_create1(Utilities::get_underlying_value<EpollFlags>(EpollFlag)),
    "create new epoll instance (::epoll_create())");
}


template <EpollFlags EpollFlag>
template <ControlOperations ControlOperation>
int Epoll<EpollFlag>::control_operation(
  const int fd,
  const EpollEvent& epoll_event)
{
  int result {
    CheckReturn()(
      ::epoll_ctl(
        fd_,
        static_cast<int>(ControlOperation),
        fd,
        epoll_event.to_epoll_event()),
      "Failed to perform control operation on either " + std::to_string(fd_) +
        " or " + std::to_string(fd))};

  last_epoll_event_ = epoll_event;

  return result;  
}

template <EpollFlags EpollFlag>
template <
  int MaxEvents,
  int Timeout,
  typename
  >
int Epoll<EpollFlag>::wait(const EpollEvent& events)
{
  int result {
    CheckWait()(::epoll_wait(fd_, events.to_epoll_event(), MaxEvents, Timeout))};

  last_epoll_event_ = events;

  return result;
}

} // namespace IO

#endif // _IO_EPOLL_H_
