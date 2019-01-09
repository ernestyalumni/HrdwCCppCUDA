//------------------------------------------------------------------------------
/// \file POSIXElapsedTime.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Measures elapsed time from POSIX Linux system calls.
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
#ifndef _TIME_POSIX_ELAPSED_TIME_H_
#define _TIME_POSIX_ELAPSED_TIME_H_

#include "Clocks.h" // ClockIds, GetClockTime
#include "Specifications.h" // TimeSpecification

namespace Time
{

//------------------------------------------------------------------------------
/// \brief Use TimeSpecifications and GetClockTime
//------------------------------------------------------------------------------
template <ClockIds ClockId = ClockIds::monotonic>
class POSIXElapsedTime
{
  public:

    POSIXElapsedTime():
      t_0_{},
      get_clock_time_{}
    {}

    void start()
    {
      t_0_ = get_clock_time_();
    }

    TimeSpecification operator()()
    {
      return get_clock_time_() - t_0_;
    }

    TimeSpecification t_0() const
    {
      return t_0_;
    }

    GetClockTime<ClockId> get_clock_time() const
    {
      return get_clock_time_;
    }

  private:

    TimeSpecification t_0_;

    GetClockTime<ClockId> get_clock_time_;
}; // class POSIXElapsedTime

} // namespace Time

#endif // _TIME_POSIX_ELAPSED_TIME_H_
