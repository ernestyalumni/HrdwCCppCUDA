
//------------------------------------------------------------------------------
/// \file Epoll.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  An epoll instance as RAII 
/// \ref      
/// \details Using RAII for an epoll instance. 
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
///  g++ -std=c++14 Socket_main.cpp -o Socket_main
//------------------------------------------------------------------------------
#ifndef _EPOLL_H_
#define _EPOLL_H_

#include <algorithm> // std::for_each
#include <array>
#include <cerrno> // errno
#include <cstring> // std::strerror
#include <iostream> // std::cerr
#include <stdexcept> // std::out_of_range
#include <sys/epoll.h> // for epoll_create1(), epoll_ctl(), struct epoll_event
#include <system_error>
#include <unistd.h> // for close(), read()
#include <utility> // std::pair, std::get
#include <vector>

namespace IO
{

constexpr int MAX_EVENTS {5};   // maximum number of events for ::epoll_wait().
constexpr int TIMEOUT {30000}; // 30000 ms or 30 secs.

//------------------------------------------------------------------------------
/// \brief enum class for all event types.
/// \details Events member of epoll_event is a bit mask composed by ORing
/// together 0 or more of the following available event types.
/// \ref http://man7.org/linux/man-pages/man2/epoll_ctl.2.html
//------------------------------------------------------------------------------
enum class EventTypes : uint32_t
{
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
//template <uint32_t EventType>
class EpollEvent : public ::epoll_event
{
  public:

//    EpollEvent():
  //    ::epoll_event{EventType, {}}
    //{}
    EpollEvent() = default;

    EpollEvent(const uint32_t event):
      ::epoll_event{event, {}}
    {}

    EpollEvent(const EventTypes event_type):
      EpollEvent{
        static_cast<std::underlying_type_t<EventTypes>>(event_type)}
    {}

    // \ref https://en.cppreference.com/w/c/language/struct_initialization
    // \ref https://en.cppreference.com/w/cpp/language/aggregate_initialization
    EpollEvent(const uint32_t event, const int fd):
      ::epoll_event{event, {.fd = fd}}
    {}

    EpollEvent(const EventTypes event_type, const int fd):
      EpollEvent{
        static_cast<std::underlying_type_t<EventTypes>>(event_type),
        fd}
    {}

    // Consider using &epoll_event for Epollevent epoll_event, instead.
    ::epoll_event* to_epoll_event()
    {
      return reinterpret_cast<::epoll_event*>(this);
    }
};

//------------------------------------------------------------------------------
/// \brief enum class for all event types.
/// \details Events member of epoll_event is a bit mask composed by ORing
/// together 0 or more of the following available event types.
/// \ref http://man7.org/linux/man-pages/man2/epoll_ctl.2.html
//------------------------------------------------------------------------------
enum class ControlOperation : int
{
  add = EPOLL_CTL_ADD, // register target `fd` on `epoll` instance, `epfd`.
  modify = EPOLL_CTL_MOD, // change event `event` associated with `fd`
  remove = EPOLL_CTL_DEL // remove (degister) target fd from epoll instance.
};

//------------------------------------------------------------------------------
/// \brief An epoll instance class template.
/// \details Create an epoll instance with a std::vector full of fds.
/// Flags is 0 or `EPOLL_CLOEXEC`.
//------------------------------------------------------------------------------
template <int Flags = 0, unsigned int MaxEvents = MAX_EVENTS>
class Epoll : public std::vector<std::pair<int, EpollEvent>>
{
  public:

    using FdEventPair = std::pair<int, EpollEvent>;

    Epoll():
      epoll_fd_{create_an_instance()}
    {}

    ~Epoll()
    {
      if (::close(epoll_fd_) < 0)
      {
        std::cerr << 
          "Failed to close file descriptor, epoll_fd_ {::close} : " <<
          epoll_fd_ << '\n';
      }
    }

    template <int Op>
    void control_fd(const int fd, EpollEvent& epoll_event)
    {
      if (::epoll_ctl(epoll_fd_, Op, fd, &epoll_event) < 0)
      {
        throw std::system_error(
          errno,
          std::generic_category(),
          "Failed on ::epoll_ctl \n");
      }
    }

    template <uint32_t EventType>
    void add_fd_to_poll(const int fd)
    {
      EpollEvent epoll_event{EventType, fd};
      control_fd<static_cast<int>(ControlOperation::add)>(fd, epoll_event);
      emplace_back(std::pair<const int, EpollEvent>{fd, epoll_event});
    }

    template <int TimeOut = TIMEOUT>
    int ready_and_wait()
    {
      int epoll_wait_result {
        ::epoll_wait(epoll_fd_, events_.data(), MaxEvents, TimeOut)
      };

      if (epoll_wait_result < 0)
      {
        std::cout << " errno : " << std::strerror(errno) << '\n';
        throw std::system_error(
          errno,
          std::generic_category(),
          "Error occurred for ::epoll_wait, on epoll_fd : " + 
            std::to_string(epoll_fd_));
      }
      return epoll_wait_result;
    }

  protected:

    int epoll_fd() const
    {
      return epoll_fd_;
    }

    void close_all_fds()
    {
      std::for_each(
        begin(),
        end(),
        [](FdEventPair& fd_event_pair) 
          {
            if (::close(std::get<int>(fd_event_pair)) < 0)
            {
              std::cout << " errno : " << std::strerror(errno) << '\n';
              throw std::system_error(
                errno,
                std::generic_category(),
                "Failed to close file descriptor, fd_ {::close} : " +
                  std::to_string(std::get<int>(fd_event_pair)));
            }
          });
    }

    // Accessors
    int events_fd(const unsigned int i) const
    {
      if (i >= MaxEvents)
      {
        throw std::out_of_range("Out of range: index i: " + std::to_string(i));
      }
      return events_[i].data.fd;      
    }

    void print_all() const
    {
      auto print = [](const FdEventPair& fd_event_pair) {
        std::cout << " fd : " << std::get<int>(fd_event_pair) << 
          " events : " << std::get<1>(fd_event_pair).events << " data.fd : " <<
          std::get<1>(fd_event_pair).data.fd << " data.u32 : " << 
          std::get<1>(fd_event_pair).data.u32 << " data.u64 : " <<
          std::get<1>(fd_event_pair).data.u64 << '\n';
      };

      std::for_each(begin(), end(), print);
    }


  private:

    int create_an_instance()
    {
      const int epoll_fd {::epoll_create1(Flags)};
      if (epoll_fd < 0)
      {
        throw std::runtime_error(
          "Failed to create epoll instance, ::epoll_create1");
      }
      else
      {
        return epoll_fd;
      }
    }

    int epoll_fd_;
    std::array<EpollEvent, MaxEvents> events_;
};

//------------------------------------------------------------------------------
/// \brief Single Epoll class.
/// \details Create an epoll instance with a single fd.
//------------------------------------------------------------------------------
template <int Flags>
class SingleEpoll
{
  public:

    explicit SingleEpoll():
      fd_{::epoll_create1(Flags)}
    {}

    ~SingleEpoll()
    {
      if (::close(fd_) < 0)
      {
        std::cerr << "Failed to close file descriptor, fd_ {::close} : " <<
          fd_ << '\n';
      }
    }

  protected:

    int fd() const
    {
      return fd_;
    }

  private:

    int fd_;
};


} // namespace Sockets

#endif // _SOCKET_H_
