//------------------------------------------------------------------------------
/// \file TimerFd.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  A POSIX clock(s).
/// \ref https://linux.die.net/man/3/clock_gettime
/// \details
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
///  g++ -std=c++14 -lrt ../MessageQueue_main.cpp -o ../MessageQueue_main
//------------------------------------------------------------------------------
#ifndef _UTILITIES_TIMER_FD_H_
#define _UTILITIES_TIMER_FD_H_

#include "Clocks.h" // ClockIDs, check_result

#include <iostream>
#include <sys/timerfd.h>
#include <unistd.h> // ::close

namespace Utilities
{

//------------------------------------------------------------------------------
/// \brief enum class for all clock flags, that maybe bitwise ORed in to change
/// behavior of timerfd_create().
///
//------------------------------------------------------------------------------
enum class ClockFlags : int
{
  default_value = 0,
  non_blocking = TFD_NONBLOCK,
  close_on_execute = TFD_CLOEXEC
};


//------------------------------------------------------------------------------
/// \brief enum class for all set time flags,
/// \details The `flags` argument is either 0, to start a relative timer
/// (`new_value.it_value` specifies a time relative to current value of the
/// clock specified by `clockid`), or `TFD_TIMER_ABSTIME`, to start an absolute
/// timer (`new_value.it_value` specifies an absolute time for the clock
/// specified by `clockid`; i.e. timer will expire when value of that clock
/// reaches value specified in `new_value.it_value`.)
/// \ref https://linux.die.net/man/2/timerfd_settime
//------------------------------------------------------------------------------
enum class SetTimeFlags : int
{
  default_value = 0,
  absolute_time = TFD_TIMER_ABSTIME//,
//  cancel_on_set = TFD_TIMER_CANCEL_ON_SET
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
    ::itimerspec{{0, 0}, {0, 0}}
  {}

  //----------------------------------------------------------------------------
  /// \brief Constructor by type Duration for a periodic interval and initial
  /// expiration.
  /// \details `new_value.it_interval` - set 1 or both fields to nonzero values
  /// specifies the period, in seconds and nanoseconds, for repeated timer
  /// expirations after initial expiration.
  /// If both fields of `new_value.it_interval` are 0, timer expires just once,
  /// at time specified by `new_value.it_value`
  ///
  /// `new_value.it_value` - specifies initial expiration of timer, in
  /// seconds and nanoseconds.
  /// Setting either field of `new_value.it_value` to nonzero value arms timer.
  /// Setting both fields of `new_value.it_value` to 0 disarms timer.
  //----------------------------------------------------------------------------
  template <
    class Duration,
    typename = std::enable_if_t<
      std::is_class<Duration>::value &&
      std::is_compound<Duration>::value>
    >
  IntervalTimerSpecification(
    const Duration& initial_expiration,
    const Duration& interval
    ):
    ::itimerspec{as_timespec(interval), as_timespec(initial_expiration)}
  {}

  template <
    class Duration,
    typename = std::enable_if_t<
      std::is_class<Duration>::value &&
      std::is_compound<Duration>::value>
    >
  explicit IntervalTimerSpecification(const Duration& initial_expiration):
    IntervalTimerSpecification{initial_expiration, Duration::zero()}
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

  template <class Duration>
  ::timespec as_timespec(const Duration& duration)
  {
    const Seconds duration_secs {duration_cast<Seconds>(duration)};
    const Nanoseconds duration_nanosecs {
      duration_cast<Nanoseconds>(duration - duration_secs)};

    return ::timespec{duration_secs.count(), duration_nanosecs.count()};
  }
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
  ClockFlags clock_flags = ClockFlags::default_value>
class TimerFd
{
  public:

    template <
      class Duration,
      typename = std::enable_if_t<
        std::is_class<Duration>::value &&
        std::is_compound<Duration>::value>
      >
    TimerFd(
      const Duration& initial_expiration,
      const Duration& interval
      ):
      fd_{::timerfd_create(
        static_cast<int>(clock_id),
        static_cast<int>(clock_flags))},
      new_value_{IntervalTimerSpecification{initial_expiration, interval}},
      old_value_{IntervalTimerSpecification{}},
      current_value_{IntervalTimerSpecification{}},
      expirations_buffer_{0}
    {
      check_result(
        fd_,
        "create file descriptor (::timerfd_create)");
    }

    template <
      class Duration,
      typename = std::enable_if_t<
        std::is_class<Duration>::value &&
        std::is_compound<Duration>::value>
      >
    explicit TimerFd(const Duration& initial_expiration):
      TimerFd{initial_expiration, Duration::zero()}
    {}

    TimerFd() = delete;

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

    IntervalTimerSpecification current_value() const
    {
      return current_value_;
    }

    const uint64_t expirations() const
    {
      return expirations_buffer_;
    }


    template <
      SetTimeFlags flag = SetTimeFlags::default_value,
//      int flag = 0,
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
          static_cast<int>(flag),
//          flag,
          new_value_.to_itimerspec(),
          old_value_.to_itimerspec());
      }
      else
      {
        set_time_results =
          ::timerfd_settime(
          fd_,
          static_cast<int>(flag),
//          flag,
          new_value_.to_itimerspec(),
          nullptr);
      }

      check_result(set_time_results, "set time (::timerfd_settime)");
      return set_time_results;
    }

    /*
    int set_time(const int flags = 0, const bool provide_old_value = false)
    {
      int set_time_results;

      // since old_value argument is not NULL (nullptr), then old_value used to
      // return setting of the timer that was current at the time of the call.
      if (provide_old_value)
      {
        set_time_results =
          ::timerfd_settime(
            fd_,
            flags,
            new_value_.to_itimerspec(),
            old_value_.to_itimerspec());
      }
      else
      {
        set_time_results =
          ::timerfd_settime(
            fd_,
            flags,
            new_value_.to_itimerspec(),
            nullptr);
      }

      check_result(set_time_results, "set time (::timerfd_settime)");
      return set_time_results;
    }
    */

    // returns current setting of the timer referred to by fd.
    // it_interval field returns interval of the timer. If both fields of this
    // structure are 0, then timer set to expire just once
    IntervalTimerSpecification get_time()
    {
      const int get_time_results {
        ::timerfd_gettime(fd_, current_value_.to_itimerspec())};
      check_result(
        get_time_results,
        "get time (::timerfd_gettime)");
      return current_value_;
    }

    // returns current setting of the timer referred to by fd.
    // it_interval field returns interval of the timer. If both fields of this
    // structure are 0, then timer set to expire just once
    // it_value field returns amount of time until timer will next expire.
    int get_time(IntervalTimerSpecification& its)
    {
      const int get_time_results {
        ::timerfd_gettime(fd_, its.to_itimerspec())};
      check_result(
        get_time_results,
        "get time (::timerfd_gettime)");
      return get_time_results;
    }

    template <ClockIDs a_clock_id, ClockFlags some_clock_flags>
    friend std::ostream& operator<<(std::ostream& os,
      const TimerFd<a_clock_id, some_clock_flags>& tfd);

    // \brief Assigns unsigned 8-byte integer containing number of expirations
    // to class data member expirations_buffer_.
    void read()
    {
      expirations_buffer_ = 0;
      ssize_t read_result {
        ::read(fd_, &expirations_buffer_, sizeof(uint64_t))};
      if (read_result != sizeof(uint64_t))
      {
        std::cout << " errno : " << std::strerror(errno) << '\n';
//        throw std::system_error(
  //        errno,
    //      std::generic_category(),
      //    "Failed to read from fd (::read)\n");
      }
    }

  private:

    int fd_;
    IntervalTimerSpecification new_value_;
    IntervalTimerSpecification old_value_;
    IntervalTimerSpecification current_value_;
    uint64_t expirations_buffer_;
};

template <ClockIDs a_clock_id, ClockFlags some_clock_flags>
std::ostream& operator<<(std::ostream& os,
  const TimerFd<a_clock_id, some_clock_flags>& tfd)
{
  os << "fd : " << tfd.fd_ << " new_value: " << tfd.new_value() << "old_value: " <<
    tfd.old_value();

  return os;
}

} // namespace Utilities

#endif // _UTILITIES_CLOCK_H_
