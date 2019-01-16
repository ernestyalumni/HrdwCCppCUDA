//------------------------------------------------------------------------------
/// \file Timer.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  2-Tuple.
/// \ref Ch. 21 Class Hierarchies, 21.2.Design of Class Hierarchies
///   The C++ Programming Language, 4th Ed., Stroustrup;
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
///     Timer_main.cpp -o Timer_main
//------------------------------------------------------------------------------
#ifndef _TIME_TIMER_H_
#define _TIME_TIMER_H_

#include "Clock.h" // Clock, StdSteadyClock
#include "Specifications.h"
#include "Utilities/Chrono.h" // duration_cast

#include <stdexcept> // std::invalid_argument

namespace Time
{

//------------------------------------------------------------------------------
/// \class Timer
/// \brief Useful interface for a timer.
/// \url https://en.wikipedia.org/wiki/Timer
/// \details Provide a useful API that one would expect to be able to use in a
/// full-fledged timer, including the ability to
///   * start (and restart if used more than once) a timer,
///   * provide the elapsed time from last start (i.e. dual stopwatch ability),
///   * get back the fixed expiration time (the amount of time from the last
/// start, to when we consider the timer expired) for convenience,
///   * countdown time (time until expiration),
///   * whether the time interval that the timer had been set to had expired
//------------------------------------------------------------------------------
template <class Duration>
class Timer
{
  public:

    // Data is gone; ctors gone since there's no data to initialize.

    //--------------------------------------------------------------------------
    /// \fn start
    //--------------------------------------------------------------------------
    virtual void start() = 0; // pure virtual function

    //--------------------------------------------------------------------------
    /// \fn elapsed_time
    //--------------------------------------------------------------------------
    virtual Duration elapsed_time() const = 0; // pure virtual function

    virtual Duration expiration_time() const = 0;

    //--------------------------------------------------------------------------
    /// \fn countdown_time
    /// \detail Return the countdown time which has a range in value from and
    /// including 0 to, and including, the expiration time.
    //--------------------------------------------------------------------------
    virtual Duration countdown_time() const = 0;

    virtual bool is_expired() const = 0;

    // Add virtual destructor to ensure proper cleanup of data that'll be
    // defined in derived class
    virtual ~Timer()
    {}
}; // class Timer

template <class Duration, class TimePointType>
class ClockTimer : public Timer<Duration>, public Clock<TimePointType>
{
  public:

    virtual void start()
    {
      Clock<TimePointType>::store_current_time();
    }

    virtual Duration elapsed_time() const
    {
      return (Clock<TimePointType>::get_current_time() -
        Clock<TimePointType>::get_stored_time());
    }

    virtual Duration countdown_time() const
    {
      const Duration time_to_expire {
        Timer<Duration>::expiration_time() - Timer<Duration>::elapsed_time()};

      return ((time_to_expire < Duration{0} ? Duration{0} : time_to_expire));
    }

    virtual bool is_expired()
    {
      return (countdown_time() == Duration{0});
    }

    ~ClockTimer()
    {}
};

template <class Duration>
class StdSteadyClockTimer : public Timer<Duration>, public StdSteadyClock<Duration>
{
  public:

    StdSteadyClockTimer() = delete;

    explicit StdSteadyClockTimer(const Duration& expiration_time):
      expiration_time_{expiration_time}
    {
      start();
    }

    void start()
    {
      StdSteadyClock<Duration>::store_current_time();
    }

    Duration elapsed_time() const
    {
      return Utilities::duration_cast<Duration>(
        StdSteadyClock<Duration>::get_current_time() -
          StdSteadyClock<Duration>::get_stored_time());
    }

    Duration expiration_time() const
    {
      return expiration_time_;
    }

    Duration countdown_time() const
    {
      const Duration time_to_expire {expiration_time() - elapsed_time()};

      return ((time_to_expire < Duration{0} ? Duration{0} : time_to_expire));
    }

    bool is_expired() const
    {
      return (countdown_time() == Duration{0});
    }

  private:

    Duration expiration_time_;
};

class POSIXMonotonicClockTimer : public Timer<TimeSpecification>,
  public POSIXMonotonicClock
{
  public:

    POSIXMonotonicClockTimer() = delete;

    explicit POSIXMonotonicClockTimer(
      const TimeSpecification& expiration_time
      ):
      expiration_time_{expiration_time}
    {
      start();
    }

    void start()
    {
      POSIXMonotonicClock::store_current_time();
    }

    TimeSpecification elapsed_time() const
    {
      return (POSIXMonotonicClock::get_current_time() -
        POSIXMonotonicClock::get_stored_time());
    }

    TimeSpecification expiration_time() const
    {
      return expiration_time_;
    }

    TimeSpecification countdown_time() const
    {
      try
      {
        const TimeSpecification time_to_expire {
          expiration_time() - elapsed_time()};
        return time_to_expire;
      }
      catch (const std::invalid_argument& e)
      {
        return TimeSpecification{};
      }
    }

    bool is_expired() const
    {
      return (countdown_time() == TimeSpecification{});
    }

  private:

    TimeSpecification expiration_time_;
};

} // namespace Time

#endif // _TIME_TIMER_H_
