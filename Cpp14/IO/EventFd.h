//------------------------------------------------------------------------------
/// \file EventFd.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  A file descriptor (fd) for event notification.
/// \ref http://man7.org/linux/man-pages/man2/eventfd.2.html  
/// \details 
/// \copyright If you find this code useful, feel free to donate directly via
/// PayPal (username ernestyalumni or email address above); my PayPal profile:
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
///  g++ -std=c++14 EventFd_main.cpp -o EventFd_main
//------------------------------------------------------------------------------
#ifndef _IO_EVENT_FD_H_
#define _IO_EVENT_FD_H_

#include "../Utilities/CheckReturn.h" // CheckReturn, check_valid_fd, check_read
// check_write
#include "../Utilities/casts.h" // get_underlying value

#include <iostream>
#include <stdexcept> // std::runtime_error
#include <sys/eventfd.h>
#include <unistd.h> // ::read, ::close

namespace IO
{

//------------------------------------------------------------------------------
/// \brief enum class for all eventfd flags, that maybe bitwise ORed in to
/// change behavior of eventfd().
//------------------------------------------------------------------------------
enum class EventFdFlags : int
{
  default_value = 0,
  close_on_execute = EFD_CLOEXEC,
  non_blocking = EFD_NONBLOCK,
  semaphore = EFD_SEMAPHORE
};

//------------------------------------------------------------------------------
/// \brief fd for event notification.
/// \details Creates "eventfd object" that can be used as an event wait/notify
/// mechanism by user-space applications, and by kernel to notify user-space
/// applications of events.
//------------------------------------------------------------------------------
template <EventFdFlags flags = EventFdFlags::default_value>
class EventFd
{
  public:

    using CheckReturn = Utilities::CheckReturn;

    explicit EventFd(const int initval = 0):
      fd_{::eventfd(initval, static_cast<int>(flags))}
    {
      Utilities::check_valid_fd(fd_, "create file descriptor (::eventfd)");
    }

    // Not copyable, movable.
    EventFd(const EventFd&) = delete;
    EventFd& operator=(const EventFd&) = delete;

    EventFd(EventFd&&) = default;
    EventFd& operator=(EventFd&&) = default;

    ~EventFd()
    {
      Utilities::check_valid_fd(::close(fd_), "close fd (::close)"); 
    }

    int fd() const
    {
      return fd_;
    }

    uint64_t buffer() const
    {
      return buffer_;
    }

    void set_buffer(const uint64_t value)
    {
      buffer_ = value;
    }

    //--------------------------------------------------------------------------
    /// \brief Returns 8-byte int, in host byte order, such that
    /// if EFD_SEMAPHORE was not specified, and eventfd counter is nonzero,
    /// then read returns counter value, and counter's value reset to 0,
    /// if EFD_SEMAPHORE was specified, and eventfd counter is nonzero, then
    /// read returns 1, and counter's value decremented by 1,
    /// if counter is 0, then call either blocks until counter nonzero, (at 
    /// which time read proceeds as described above), or fails with error
    /// EAGAIN if fd is nonblocking.
    //--------------------------------------------------------------------------
    void read()
    {
      const ssize_t read_result {::read(fd_, &buffer_, sizeof(uint64_t))};

      try
      {
        Utilities::check_read<uint64_t>(read_result);
      }
      catch (const std::runtime_error& e)
      {
        std::cout << " A standard runtime error was caught, with message " <<
          e.what() << '\n';
      }
    }

    //--------------------------------------------------------------------------
    /// \brief Adds 8-byte integer value supplied in its buffer to the counter.
    /// \details If addition would cause counter's valus to exceed maximum,
    /// then write either blocks until a read is performed on fd, or fails with
    /// error EAGAIN if fd is made nonblocking.
    //--------------------------------------------------------------------------    
    void write()
    {
      const ssize_t write_result{::write(fd_, &buffer_, sizeof(uint64_t))};

      try
      {
        Utilities::check_write<sizeof(uint64_t)>(write_result);
      }
      catch (const std::runtime_error& e)
      {
        std::cout << " A standard runtime error was caught, with message " <<
          e.what() << '\n';
      }
    }

  private:

    int fd_;
    uint64_t buffer_;
};

} // namespace IO

#endif // _IO_EVENT_FD_H_
