//------------------------------------------------------------------------------
/// \file TimerFd.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  A timerfd as RAII 
/// \ref      
/// \details Using RAII for timerfd. 
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
#ifndef _IPC_TIMERFD_H_
#define _IPC_TIMERFD_H_

#include <cerrno> // errno
#include <chrono> // std::chrono
#include <cstring> // strerror
#include <iostream>
#include <sys/timerfd.h>
#include <system_error>
#include <unistd.h> // ::close

namespace IPC
{

namespace Time
{

//------------------------------------------------------------------------------
/// \brief enum class for all clock id's, type of clock.
/// \details clockid argument specifies clock that's used to mark the progress
/// of the timer
/// \ref http://man7.org/linux/man-pages/man2/timerfd_create.2.html
//------------------------------------------------------------------------------
enum class ClockIDs : int
{
  real_time = CLOCK_REALTIME,
  monotonic = CLOCK_MONOTONIC,
  boot_time = CLOCK_BOOTTIME,
  real_time_alarm = CLOCK_REALTIME_ALARM,
  boot_time_alarm = CLOCK_BOOTTIME_ALARM
};

//------------------------------------------------------------------------------
/// struct timespec
/// {
///   time_t tv_sec; // Seconds
///   long tv_nsec; // Nanoseconds
/// };
///
/// struct itimerspec
/// {
///   struct timerspec it_interval; // Interval for periodic timer
///   struct timerspec it_value; // Initial expiration
/// };
//------------------------------------------------------------------------------
struct IntervalTimerSpecification : public ::itimerspec 
{
  using Nanoseconds = std::chrono::nanoseconds;
  using Seconds = std::chrono::seconds;

  explicit IntervalTimerSpecification(
    const long periodic_timer_interval_sec,
    const long periodic_timer_interval_nsec,
    const long initial_expiration_time_sec,
    const long initial_expiration_time_nsec = 0):
    ::itimerspec{
      {periodic_timer_interval_sec, periodic_timer_interval_nsec},
      {initial_expiration_time_sec, initial_expiration_time_nsec}}
  {}

//  explicit IntervalTimerSpecification(
  //  const Seconds )

  explicit IntervalTimerSpecification(
    const long periodic_timer_interval_sec,
    const long periodic_timer_interval_nsec = 0):
    ::itimerspec{{periodic_timer_interval_sec, periodic_timer_interval_nsec}}
  {}

  IntervalTimerSpecification():
    IntervalTimerSpecification{0, 0}
  {}

  const ::itimerspec* to_itimerspec() const
  {
    return reinterpret_cast<const ::itimerspec*>(this);
  }

  ::itimerspec* to_itimerspec()
  {
    return reinterpret_cast<::itimerspec*>(this);
  }

  friend std::ostream& operator<<(std::ostream& os,
    const IntervalTimerSpecification& its);
};

std::ostream& operator<<(std::ostream& os,
  const IntervalTimerSpecification& its)
{
  os << its.it_interval.tv_sec << ' ' << its.it_interval.tv_nsec << ' ' <<
    its.it_value.tv_sec << ' ' << its.it_value.tv_nsec << '\n';

  return os;
}

template <
  ClockIDs clock_id = ClockIDs::monotonic,
  int clock_flags = TFD_NONBLOCK>
class TimerFd
{
  public:

    explicit TimerFd(const IntervalTimerSpecification& new_its,
      const IntervalTimerSpecification& old_its):
      fd_{::timerfd_create(static_cast<int>(clock_id), clock_flags)},
      new_value_{new_its},
      old_value_{old_its}
    {
      check_result(fd_, "create file descriptor (::timerfd_create)");      
    }

    explicit TimerFd(const IntervalTimerSpecification& new_its):
      TimerFd{new_its, IntervalTimerSpecification{0, 0, 0, 0}}
    {}

    TimerFd():
      TimerFd{
        IntervalTimerSpecification{0, 0, 0, 0},
        IntervalTimerSpecification{0, 0, 0, 0}}
    {}

    ~TimerFd()
    {
      check_result(::close(fd_), "close fd (::close)");
    }

    // Accessors and Getters
    IntervalTimerSpecification new_value() const
    {
      return new_value_;
    }

    void new_value(const IntervalTimerSpecification& its)
    {
      new_value_ = its;
    }

    IntervalTimerSpecification old_value() const
    {
      return old_value_;
    }

    void old_value(const IntervalTimerSpecification& its)
    {
      old_value_ = its;
    }

    #if 0
    IntervalTimerSpecification current_value() const
    {
      return current_value_;
    }
    #endif 

    template <
      int flag = TFD_TIMER_ABSTIME,
      bool provide_old_value = true>
    int set_time()
    {
      int set_time_results;

      // since old_value argument is not NULL (nullptr), then old_value used to
      // return setting of the timer that was current at the time of the call.
      if (provide_old_value)
      {
        set_time_results =
          ::timerfd_settime(
          fd_,
          flag,
          new_value_.to_itimerspec(),
          old_value_.to_itimerspec());
      }
      else
      {
        set_time_results =
          ::timerfd_settime(
          fd_,
          flag,
          new_value_.to_itimerspec(),
          nullptr);
      }

      check_result(set_time_results, "set time (::timerfd_settime)");
      return set_time_results;
    }

    // returns current setting of the timer referred to by fd.
    // it_interval field returns interval of the timer. If both fields of this
    // structure are 0, then timer set to expire just once
    int get_time()
    {
      const int get_time_results {
        ::timerfd_gettime(fd_, current_value_.to_itimerspec())};
      check_result(get_time_results, "get time (::timerfd_gettime)");
      return get_time_results;
    }

    // returns current setting of the timer referred to by fd.
    // it_interval field returns interval of the timer. If both fields of this
    // structure are 0, then timer set to expire just once
    // it_value field returns amount of time until timer will next expire.
    int get_time(IntervalTimerSpecification& its)
    {
      const int get_time_results {
        ::timerfd_gettime(fd_, its.to_itimerspec())};
      check_result(get_time_results, "get time (::timerfd_gettime)");
      return get_time_results;
    }

    template <ClockIDs a_clock_id, int some_clock_flags>
    friend std::ostream& operator<<(std::ostream& os,
      const TimerFd<a_clock_id, some_clock_flags>& tfd);

    // returns unsigned 8-byte integer containing number of expirations.
    uint64_t read() const
    {
      uint64_t buffer;
      ssize_t read_result {
        ::read(fd_, &buffer, sizeof(uint64_t))};
      if (read_result != sizeof(uint64_t))
      {
        std::cout << " errno : " << std::strerror(errno) << '\n';
        throw std::system_error(
          errno,
          std::generic_category(),
          "Failed to read from fd (::read)\n");        
      }
      return buffer;
    }


  private:

    // Specified by Linux manual page.
    int check_result(int e, const std::string& custom_error_string)
    {
      if (e < 0)
      {
        std::cout << " errno : " << std::strerror(errno) << '\n';
        throw std::system_error(
          errno,
          std::generic_category(),
          "Failed to " + custom_error_string + "\n");
      }
      return errno;
    }

    int fd_;
    IntervalTimerSpecification new_value_;
    IntervalTimerSpecification old_value_;
    IntervalTimerSpecification current_value_;
};


template <ClockIDs a_clock_id, int some_clock_flags>
std::ostream& operator<<(std::ostream& os,
  const TimerFd<a_clock_id, some_clock_flags>& tfd)
{
  os << "fd : " << tfd.fd_ << '\n' << tfd.new_value() << "old_value: \n" <<
    tfd.old_value();

  return os;
}


} // namespace Time

} // namespace IPC

#endif // _IPC_TIMERFD_H_
