//------------------------------------------------------------------------------
/// \file Chrono.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Date and time utilities.
/// \ref https://en.cppreference.com/w/cpp/chrono
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
#ifndef _UTILITIES_CHRONO_H_
#define _UTILITIES_CHRONO_H_

#include <chrono>

namespace Utilities
{

//------------------------------------------------------------------------------
/// \brief Clocks - consists of a starting point (or epoch) and tick rate.
/// \ref https://en.cppreference.com/w/cpp/chrono/
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
/// \brief Wall clock time from syste-wide realtime clock
/// \details Only C++ clock that has ability to map its time points to C-style
/// time, and, therefore, to be displayed.
/// \ref https://en.cppreference.com/w/cpp/chrono/
//------------------------------------------------------------------------------
using SystemClock = std::chrono::system_clock;

// Since C++20
template <class Duration>
using SystemTime = std::chrono::time_point<SystemClock, Duration>;

//------------------------------------------------------------------------------
/// \brief Monotonic clock.
/// \details Not related to wall clock time; most suitable for measuring
/// intervals.
/// \ref https://en.cppreference.com/w/cpp/chrono/
//------------------------------------------------------------------------------
using SteadyClock = std::chrono::steady_clock;

template <class Duration>
using SteadyTimePoint = std::chrono::time_point<SteadyClock, Duration>;


//------------------------------------------------------------------------------
/// \brief Monotonic clock that'll never be adjusted.
/// \ref https://en.cppreference.com/w/cpp/chrono/
//------------------------------------------------------------------------------
using SteadyClock = std::chrono::steady_clock;

using HighResolutionClock = std::chrono::high_resolution_clock;

//------------------------------------------------------------------------------
/// \brief Class template specializations that represents a time interval.
/// \details For std::chrono::duration
/// \ref https://en.cppreference.com/w/cpp/chrono/duration
//------------------------------------------------------------------------------
using Nanoseconds = std::chrono::nanoseconds;
using Microseconds = std::chrono::microseconds;
using Milliseconds = std::chrono::milliseconds;
using Seconds = std::chrono::seconds;

//------------------------------------------------------------------------------
/// \brief Function template that converts to ToDuration.
/// \details For std::chrono::duration_cast
/// \ref https://en.cppreference.com/w/cpp/chrono/duration/duration_cast
/// https://stackoverflow.com/questions/26184190/alias-a-templated-function
//------------------------------------------------------------------------------
template <class ToDuration, class T>
constexpr ToDuration duration_cast(const T& d)
{
  return std::chrono::duration_cast<ToDuration>(d);
}

namespace Literals
{

// Defined in inline namespace std::literals::chrono_literals
using namespace std::chrono_literals;

} // namespace Literals

template <typename Duration = Milliseconds, class Clock = SteadyClock>
class ElapsedTime
{
  public:

    ElapsedTime():
      t_i_{
        std::chrono::time_point_cast<Duration>(Clock::now())}
    {}

    Duration operator()()
    {
      const std::chrono::time_point<Clock, Duration> t_f {
        std::chrono::time_point_cast<Duration>(Clock::now())};
      return duration_cast<Duration>(t_f - t_i_);
    }

    void reset()
    {
      t_i_ = std::chrono::time_point_cast<Duration>(Clock::now());
    }

  private:

    std::chrono::time_point<Clock, Duration> t_i_;
};


} // namespace Utilities

#endif // _UTILITIES_CHRONO_H_
