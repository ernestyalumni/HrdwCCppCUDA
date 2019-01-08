//------------------------------------------------------------------------------
/// \file TimerFd.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  A POSIX timer that delivers expirations by file descriptor (fd).
/// \ref http://man7.org/linux/man-pages/man2/timerfd_create.2.html
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
///  g++ -std=c++17 -I ../ ../Utilities/Errno.cpp \
///   ../Utilities/ErrorHandling.cpp Specifications.cpp Clocks.cpp \
///     TimerFd_main.cpp -o TimerFd_main
//------------------------------------------------------------------------------
#ifndef _TIME_TIMER_FD_H_
#define _TIME_TIMER_FD_H_

#include "Clocks.h" // ClockIDs, check_result
#include "Specifications.h" // IntervalTimerSpecification
#include "Utilities/ErrorHandling.h"
#include "Utilities/casts.h" // get_underlying_value

#include <ostream>
#include <sys/timerfd.h>
#include <type_traits> // std::underlying_type_t
#include <unistd.h>

namespace Time
{

//------------------------------------------------------------------------------
/// \brief enum class for all clock flags, that maybe bitwise ORed in to change
/// behavior of timerfd_create().
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

template <
  ClockIds ClockId = ClockIds::monotonic,
  int clock_flags =
    static_cast<std::underlying_type_t<ClockFlags>>(
      ClockFlags::default_value),
  int SetFlags =
    static_cast<std::underlying_type_t<SetTimeFlags>>(
      SetTimeFlags::default_value)
  >
class TimerFd
{
  public:

    TimerFd():
      fd_{::timerfd_create(
        Utilities::get_underlying_value<ClockIds>(ClockId),
        clock_flags)},
      expirations_buffer_{0},
      new_value_{},
      old_value_{},
      current_value_{}
    {
      Utilities::ErrorHandling::HandleReturnValue{errno}(
        fd_,
        "create file descriptor (::timerfd_create)");

      set_time<SetFlags>();
    }

    explicit TimerFd(
      const IntervalTimerSpecification& interval_timer_specification):
      fd_{::timerfd_create(
        Utilities::get_underlying_value<ClockIds>(ClockId),
        clock_flags)},
      expirations_buffer_{0},
      new_value_{interval_timer_specification},
      old_value_{},
      current_value_{}
    {
      Utilities::ErrorHandling::HandleReturnValue{errno}(
        fd_,
        "create file descriptor (::timerfd_create)");

      set_time<SetFlags>();
    }

    ~TimerFd()
    {
      Utilities::ErrorHandling::HandleClose{}(::close(fd_));
    }

    template <
      int Flags = static_cast<std::underlying_type_t<SetTimeFlags>>(
        SetTimeFlags::default_value),
      bool ProvideOldValue = true>
    void set_time(const IntervalTimerSpecification& new_value)
    {
      new_value_ = new_value;
      set_time<Flags, ProvideOldValue>();
    }

    //--------------------------------------------------------------------------
    /// \brief Wraps ::timerfd_gettime
    /// \details Gets current setting of the timer referred to by fd.
    /// it_interval field returns interval of the timer. If both fields of this
    /// structure are 0, then timer set to expire just once
    /// it_value field returns amount of time until timer will next expire.
    //--------------------------------------------------------------------------
    const IntervalTimerSpecification get_time()
    {
      const int get_time_results {
        ::timerfd_gettime(fd_, current_value_.to_itimerspec_pointer())};

      Utilities::ErrorHandling::HandleReturnValue{errno}(
        get_time_results, "get time (::timerfd_gettime)");

      return current_value();
    }

    //--------------------------------------------------------------------------
    /// \brief Assigns unsigned 8-byte integer containing number of expirations
    /// to class data member expirations_buffer_.
    //--------------------------------------------------------------------------
    void read()
    {
      expirations_buffer_ = 0;

      const ssize_t read_result {
        ::read(fd_, &expirations_buffer_, sizeof(uint64_t))};

      HandleReadOnTimerFd()(read_result);
    }

    // Accessors

    const uint64_t expirations() const
    {
      return expirations_buffer_;
    }

    IntervalTimerSpecification new_value() const
    {
      return new_value_;
    }

    IntervalTimerSpecification old_value() const
    {
      return old_value_;
    }

    IntervalTimerSpecification current_value() const
    {
      return current_value_;
    }

    template <ClockIds AClockId, int some_clock_flags>
    friend std::ostream& operator<<(std::ostream& os,
      const TimerFd<AClockId, some_clock_flags>& tfd);

  private:

    using HandleRead = Utilities::ErrorHandling::HandleRead;

    //--------------------------------------------------------------------------
    /// \url http://man7.org/linux/man-pages/man2/timerfd_create.2.html
    /// \details If no timer expirations have occurred at time of read, then
    /// call either blocks until next timer expiration, or fails with error
    /// EAGAIN, if fd has been made nonblocking.
    /// read fails with error EINVAL if size of supplied buffer less than 8
    /// bytes.
    /// If associated clock is either CLOCK_REALTIME or CLOCK_REALTIME_ALARM,
    /// timer is absolute (TFD_TIMER_ABSTIME), and flag TFD_TIMER_CANCEL_ON_SET
    /// was specified when calling timerfd_settime(), then read fails with
    /// error ECANCELED if real-time clock undergoes discontinuous change.
    //--------------------------------------------------------------------------
    class HandleReadOnTimerFd : public HandleRead
    {
      public:

        HandleReadOnTimerFd() = default;

        void operator()(const ssize_t number_of_bytes)
        {
          // TODO: handle EAGAIN, ECANCELED.
          //get_error_number();
          //if error_number().error_number()

          HandleRead::operator()(number_of_bytes);
        }
    };

    template <
      int Flags = static_cast<std::underlying_type_t<SetTimeFlags>>(
        SetTimeFlags::default_value),
      bool ProvideOldValue = true>
    void set_time()
    {
      int set_time_results;

      // since old_value argument is not NULL (nullptr), then old_value used to
      // return setting of the timer that was current at the time of the call.
      if (ProvideOldValue)
      {
        set_time_results =
          ::timerfd_settime(
          fd_,
          Flags,
          new_value_.to_itimerspec_pointer(),
          old_value_.to_itimerspec_pointer());
      }
      else
      {
        set_time_results =
          ::timerfd_settime(
          fd_,
          Flags,
          new_value_.to_itimerspec_pointer(),
          nullptr);
      }

      Utilities::ErrorHandling::HandleReturnValue{errno}(
        set_time_results, "set time (::timerfd_settime)");
    }

    int fd_;

    uint64_t expirations_buffer_;

    IntervalTimerSpecification new_value_;
    IntervalTimerSpecification old_value_;
    IntervalTimerSpecification current_value_;
};

template <ClockIds ClockId, int clock_flags>
std::ostream& operator<<(std::ostream& os,
  const TimerFd<ClockId, clock_flags>& tfd)
{
  os << "fd : " << tfd.fd_ << '\n' << tfd.new_value() << "old_value: \n" <<
    tfd.old_value();

  return os;
}

} // namespace Time

#endif // _TIME_TIMER_FD_H_
