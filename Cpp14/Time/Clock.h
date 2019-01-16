//------------------------------------------------------------------------------
/// \file Clock.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Interface as abstract class for concrete clock implementations.
/// \ref Ch. 21 Class Hierarchies, 21.2.Design of Class Hierarchies
///   The C++ Programming Language, 4th Ed., Stroustrup;
/// \url https://en.cppreference.com/w/cpp/chrono/time_point/time_point_cast
/// \details Class hierarchy for Interface Inheritance for a Timer
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
///     Clock_main.cpp -o Clock_main
//------------------------------------------------------------------------------
#ifndef _TIME_CLOCK_H_
#define _TIME_CLOCK_H_

#include "Clocks.h" // GetClockTime
#include "Specifications.h"
#include "Utilities/Chrono.h" // SteadyClock, SteadyTimePoint

#include <chrono>
#include <ostream>

namespace Time
{

//------------------------------------------------------------------------------
/// \class Clock
/// \brief Useful interface for watching a clock.
/// \ref https://stackoverflow.com/questions/21615970/converting-time-point-to-specific-duration-with-chrono
/// \details Consider this example of using std::chrono library with its
/// time point vs. duration.
/// std::chrono::duration and std::chrono::time_point are different types.
/// A single time_point cannot be cast into a duration. A duration is the time
/// between 2 time_points.
/// What is desired as a template argument is some class representing the units.
//------------------------------------------------------------------------------
template <class TimePointType>
class Clock
{
  public:

    // Data is gone; ctors gone since there's no data to initialize.

    //--------------------------------------------------------------------------
    /// \fn get_current_time
    /// \details Gets the current time.
    //--------------------------------------------------------------------------
    virtual TimePointType get_current_time() const = 0; // pure virtual function

    //--------------------------------------------------------------------------
    /// \fn store_current_time
    //--------------------------------------------------------------------------
    virtual void store_current_time() = 0; // pure virtual function

    //--------------------------------------------------------------------------
    /// \fn get_stored_time
    /// \brief Return the time that was last stored.
    //--------------------------------------------------------------------------
    virtual TimePointType get_stored_time() const = 0; // pure virtual function

    // Add virtual destructor to ensure proper cleanup of data that'll be
    // defined in derived class
    virtual ~Clock()
    {}

}; // class Clock

template <class Duration>
class StdSteadyClock : public Clock<Utilities::SteadyTimePoint<Duration>>
{
  public:

    StdSteadyClock():
      stored_time_{
        std::chrono::time_point_cast<Duration>(Utilities::SteadyClock::now())}
    {}

    Utilities::SteadyTimePoint<Duration> get_current_time() const
    {
      return std::chrono::time_point_cast<Duration>(
        Utilities::SteadyClock::now());
    }

    void store_current_time()
    {
      stored_time_ = get_current_time();
    }

    Utilities::SteadyTimePoint<Duration> get_stored_time() const
    {
      return stored_time_;
    }

  private:

    std::chrono::time_point<Utilities::SteadyClock, Duration> stored_time_;
};

class POSIXMonotonicClock : public Clock<TimeSpecification>
{
  public:

    POSIXMonotonicClock():
      stored_time_{GetClockTime{}()}
    {}

    TimeSpecification get_current_time() const
    {
      return GetClockTime{}();
    }

    void store_current_time()
    {
      stored_time_ = get_current_time();
    }

    TimeSpecification get_stored_time() const
    {
      return stored_time_;
    }

  private:

    TimeSpecification stored_time_;
};


} // namespace Time

#endif // _TIME_CLOCK_H_
