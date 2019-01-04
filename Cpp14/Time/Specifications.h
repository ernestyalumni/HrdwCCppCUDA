//------------------------------------------------------------------------------
/// \file Specifications.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  A POSIX time specifications.
/// \ref https://linux.die.net/man/3/clock_gettime
/// http://pubs.opengroup.org/onlinepubs/7908799/xsh/time.h.html
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
///  g++ -std=c++17 -I ../ Specifications_main.cpp -o Specifications_main
//------------------------------------------------------------------------------
#ifndef _TIME_SPECIFICATIONS_H_
#define _TIME_SPECIFICATIONS_H_

#include "Utilities/Chrono.h"

#include <ctime> // ::timespec
#include <ostream> // std::ostream
#include <stdexcept> // std::invalid_argument
#include <type_traits> // std::enable_if_t

namespace Time
{

//------------------------------------------------------------------------------
/// \brief ::timespec struct wrapper using composition.
/// \details struct timespec
/// {
///   time_t tv_sec; // Seconds
///   long tv_nsec; // Nanoseconds
/// };
/// Use composition, over inheritance, to wrap
//------------------------------------------------------------------------------
class TimeSpecification
{
  public:

    using Nanoseconds = Utilities::Nanoseconds;
    using Seconds = Utilities::Nanoseconds;

    template <
      class Duration,
      typename = std::enable_if_t<
        std::is_compound<Duration>::value && std::is_pod<Duration>::value>
    >
    explicit TimeSpecification(const Duration& duration);

    //--------------------------------------------------------------------------
    /// \brief Constructor that checks input ::timespec argument to be
    /// non-negative.
    //--------------------------------------------------------------------------
    explicit TimeSpecification(const ::timespec& timespec):
      timespec_{timespec}
    {}

    // type-conversion operator, defines a conversion to a ::timespec struct.
    operator ::timespec() const
    {
      return timespec_;
    }

    /// Accessors for Linux system call.

    const ::timespec* to_timespec_pointer() const
    {
      return &timespec_;
    }

    ::timespec* as_timespec_pointer()
    {
      reinterpret_cast<::timespec*>(this);
    }

    friend std::ostream& operator<<(std::ostream& os,
      const TimeSpecification& time_specification);

  private:

    ::timespec timespec_;
};

template <class Duration, typename>
TimeSpecification::TimeSpecification(const Duration& duration)
{
  if (duration < Duration{0})
  {
    throw std::invalid_argument("duration input argument must be non-negative");
  }

  const Seconds duration_secs {Utilities::duration_cast<Seconds>(duration)};
  const Nanoseconds duration_nanosecs {
    Utilities::duration_cast<Nanoseconds>(duration - duration_secs)};

  timespec_.tv_sec = duration_secs.count();
  timespec_.tv_nsec = duration_nanosecs.count();
}


} // namespace Time

#endif // _TIME_SPECIFICATIONS_H_