//------------------------------------------------------------------------------
/// \file PeriodicTimerFd.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  A POSIX clock(s).
/// \ref https://linux.die.net/man/3/clock_gettime     
/// \details 
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
///  g++ -std=c++14 TimerFd_main.cpp -o TimerFd_main
//------------------------------------------------------------------------------
#ifndef _UTILITIES_PERIODIC_TIMER_FD_H_
#define _UTILITIES_PERIODIC_TIMER_FD_H_

#include "Clocks.h"
#include "TimerFd.h"

#include <iostream>

#include <stdexcept> // std::runtime_error

namespace Utilities
{

template <
  ClockIDs clock_id = ClockIDs::monotonic,
  ClockFlags clock_flags = ClockFlags::default_value,
  SetTimeFlags set_time_flags = SetTimeFlags::default_value>
class PeriodicTimerFd : public TimerFd<clock_id, clock_flags>
{
  public:

    // Need default constructor for base class.
    using Utilities::TimerFd<clock_id, clock_flags>::TimerFd;
    using Utilities::TimerFd<clock_id, clock_flags>::expirations;
    using Utilities::TimerFd<clock_id, clock_flags>::get_time;
    using Utilities::TimerFd<clock_id, clock_flags>::read;
    using Utilities::TimerFd<clock_id, clock_flags>::set_time;
    using IntervalTimerSpecification = Utilities::IntervalTimerSpecification;

//    using Utilities::TimerFd<clock_id, clock_flags>::set_time<
  //    set_time_flags, true>;

    // Assume time period is the same for initial expiration and later and
    // all subsequent intervals.
    template <
      class Duration,
      typename = std::enable_if_t<
        std::is_class<Duration>::value &&
        std::is_compound<Duration>::value>
      >
    PeriodicTimerFd(const Duration& interval):
      TimerFd<clock_id, clock_flags>{interval, interval},
      expire_once_timer_fd_{interval}
    {
//      Utilities::TimerFd<clock_id, clock_flags>::set_time();
      set_time();
      expire_once_timer_fd_.set_time();
    }

    const uint64_t read_and_block()
    {
      // Get the time, before the blocking read.
      const IntervalTimerSpecification interval_timer_specification {
        expire_once_timer_fd_.get_time()};

      read();

      if (expirations() > 1)
      {
        expirations_count_ = expirations();
      }
      else if (expirations() == 1)
      {
        if ((interval_timer_specification.it_interval.tv_sec == 0) &&
          (interval_timer_specification.it_interval.tv_nsec == 0) &&
          (interval_timer_specification.it_value.tv_sec == 0) &&
          (interval_timer_specification.it_value.tv_nsec == 0))
        {
          expirations_count_ = 1;
        }
        else
        {
          expirations_count_ = 0;
        }
      }
      else
      {
        throw std::runtime_error("expirations is non-positive");
      }
      // Start up expire once timer again.
//      std::cout << expire_once_timer_fd_.new_value() << '\n';
      expire_once_timer_fd_.set_time(); 

      return expirations_count_;
    }

    const uint64_t expirations_count() const
    {
      return expirations_count_;
    }

  private:

    TimerFd<clock_id, ClockFlags::default_value> expire_once_timer_fd_;
    uint64_t expirations_count_;
};

} // namespace Utilities

#endif // _UTILITIES_CLOCK_H_
