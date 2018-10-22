//------------------------------------------------------------------------------
/// \file TimerFd_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  A timerfd as RAII 
/// \ref      
/// \details Using RAII for timerfd. 
/// \copyright If you find this code useful, feel free to donate directly and easily at 
/// this direct PayPal link: 
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
#include "Chrono.h"
#include "TimerFd.h"

#include <iostream>
#include <thread> // std::this_thread

#include <unistd.h>
#include <fcntl.h> // fcntl

using Utilities::IntervalTimerSpecification;
using Utilities::Microseconds;
using Utilities::Milliseconds;
using Utilities::Nanoseconds;
using Utilities::Seconds;
using Utilities::SetTimeFlags;
using Utilities::TimerFd;

using namespace Utilities::Literals;

int main()
{

  // IntervalTimerSpecificationConstructs
  {
    std::cout << "\n IntervalTimerSpecificationConstructs \n";

    const IntervalTimerSpecification interval_timer_specification {
      10, 13, 42, 69};
    std::cout << " interval_timer_specification : " <<
      interval_timer_specification << '\n';

    const IntervalTimerSpecification interval_timer_specification1 {10, 13};
    std::cout << " interval_timer_specification1 : " <<
      interval_timer_specification1 << '\n';

    
    const IntervalTimerSpecification interval_timer_specification2;
    std::cout << " interval_timer_specification2 : " <<
      interval_timer_specification2 << '\n';

    const IntervalTimerSpecification interval_timer_specification3 {
      2101ms, 5300123ms};
    std::cout << " interval_timer_specification3 : " <<
      interval_timer_specification3 << '\n';

    const IntervalTimerSpecification interval_timer_specification4 {
      210112345678us};
    std::cout << " interval_timer_specification4 : " <<
      interval_timer_specification4 << '\n';
    
  }

  // IntervalTimerSpecificationAccessesDataMembers
  {
    std::cout << "\n IntervalTimerSpecificationAccessesDataMembers\n";

    const IntervalTimerSpecification interval_timer_specification {
      10, 13, 42, 69};
    std::cout << " interval_timer_specification : " <<
      interval_timer_specification << '\n';

    std::cout << interval_timer_specification.it_interval.tv_sec << '\n';
    std::cout << interval_timer_specification.it_interval.tv_nsec << '\n';
    std::cout << interval_timer_specification.it_value.tv_sec << '\n';
    std::cout << interval_timer_specification.it_value.tv_nsec << '\n';

  }

  // TimerFdConstructsWithExplicitInputValues
  {
    std::cout << "\n TimerFdConstructsWithExplicitInputValues\n";
    TimerFd<> tfd {10s, 15s};
    std::cout << tfd << '\n';

    const TimerFd<> tfd1 {13ms};
    std::cout << tfd1 << '\n';
  }

  // TimerFdSetTimeWorks
  {
    std::cout << "\n TimerFdSetTimeWorks\n";
    TimerFd<> tfd {10us, 15us};
    tfd.set_time();
    std::cout << tfd << '\n'; // 0 15000 0 10000 0 0 0 0 

    tfd.new_value(IntervalTimerSpecification{5s, 8s});
    tfd.set_time<SetTimeFlags::absolute_time, false>();
    std::cout << tfd << '\n'; // 8 0 5 0 0 0 0 0
  }

  // TimerFdGetTimeGetsTimeAfterSetTime
  {
    std::cout << "\n TimerFdGetTimeGetsTimeAfterSetTime\n";
    TimerFd<> tfd {10us, 15us};
    tfd.set_time();
    tfd.get_time();
    std::cout << tfd.current_value() << '\n';
    std::this_thread::sleep_for(2us);
    tfd.get_time();
    std::cout << tfd.current_value() << '\n';

    tfd.new_value(IntervalTimerSpecification{5s, 8s});
    tfd.set_time<SetTimeFlags::absolute_time, false>();
    tfd.get_time();
    std::cout << tfd.current_value() << '\n';
    std::this_thread::sleep_for(2s);
    tfd.get_time();
    std::cout << tfd.current_value() << '\n';
  }

  // TimerFdReadsAndGivesCorrectExpirations
  {
    std::cout << "\n TimerFdReadsAndGivesCorrectExpirations\n";
    TimerFd<> tfd {5s, 5s};
    std::cout << tfd << '\n';
    tfd.set_time();
  
//    for (int delta_t {0}; delta_t < 11; ++delta_t)
    for (int delta_t {0}; delta_t < 2; ++delta_t)
    {
      std::cout << " delta_t : " << delta_t << '\n';
//      std::cout << tfd.get_time() << '\n';
      std::this_thread::sleep_for(Seconds{delta_t});
      tfd.read();
//      std::cout << tfd.get_time() << '\n';
      std::cout << "expirations() : " << tfd.expirations() << '\n';
    }    

    TimerFd<> tfd1 {250ms, 250ms};
    std::cout << tfd1 << '\n';
    tfd1.set_time();
  
    for (int delta_t {0}; delta_t < 8; ++delta_t)
    {
      std::cout << " delta_t : " << delta_t << '\n';
      std::this_thread::sleep_for(Milliseconds{125 * delta_t});
      tfd1.read();
      std::cout << "expirations() : " << tfd1.expirations() << '\n';
    }    
  }

  // TimerFdGetsTimeBeforeExpirationAndAfter
  {
    std::cout << "\n TimerFdGetsTimeBeforeExpirationAndAfter\n";
    TimerFd<> tfd {50ms, 50ms};
    std::cout << tfd.get_time() << '\n';

    std::cout << tfd << '\n';
    tfd.set_time();

    for (int delta_t {0}; delta_t < 17; ++delta_t)
    {
      std::cout << " delta_t : " << delta_t << '\n';
      std::cout << tfd.get_time() << '\n';
      std::this_thread::sleep_for(Milliseconds{10 * delta_t});
      std::cout << tfd.get_time() << '\n';
      tfd.read();
      std::cout << "expirations() : " << tfd.expirations() << '\n';
      std::cout << tfd.get_time() << '\n';
    }
  }

  // TimerFdCanExpireOnceAndGetsTimeGetsZero
  {
    std::cout << "\n TimerFdCanExpireOnceAndGetsTimeGetsZero\n";
    TimerFd<> tfd {250ms};
    std::cout << tfd.get_time() << '\n';
    std::cout << tfd << '\n';
    tfd.set_time();

    for (int delta_t {0}; delta_t < 7; ++delta_t)
    {
      TimerFd<> tfd {50ms};
      std::cout << tfd.get_time() << '\n';
      std::cout << tfd << '\n';
      tfd.set_time();

      std::cout << " delta_t : " << delta_t << '\n';
      std::cout << tfd.get_time() << '\n';
      std::this_thread::sleep_for(Milliseconds{10 * delta_t});
      std::cout << tfd.get_time() << '\n';
      tfd.read();
      std::cout << "expirations() : " << tfd.expirations() << '\n';
      std::cout << tfd.get_time() << '\n';
    }
  }
}
