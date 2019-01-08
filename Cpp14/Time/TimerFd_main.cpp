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
///  g++ -std=c++17 -I ../ ../Utilities/Errno.cpp ../Utilities/ErrorHandling.cpp Specifications.cpp Clocks.cpp TimerFd_main.cpp -o TimerFd_main
//------------------------------------------------------------------------------
#include "TimerFd.h"

#include "Specifications.h"
#include "Utilities/Chrono.h"

#include <iostream>
#include <thread> // std::this_thread
#include <type_traits> // std::underlying_type_t

using Time::ClockFlags;
using Time::IntervalTimerSpecification;
using Time::SetTimeFlags;
using Time::TimerFd;
using Utilities::Milliseconds;
using Utilities::Seconds;

using namespace Utilities::Literals;

int main()
{
  // ClockFlagsHasAllClockFlagsToChangeBehaviorOftimerfd_create
  {
    std::cout << "\n ClockFlags \n";
    std::cout << " ClockFlags::non_blocking : " <<
      static_cast<int>(ClockFlags::non_blocking) << '\n'; // 0
    std::cout << " ClockFlags::close_on_execute : " <<
      static_cast<int>(ClockFlags::close_on_execute) << '\n'; // 1
  }

  // TimerFdDefaultConstructs
  {
    std::cout << "\n TimerFdDefaultConstructs \n";
    const TimerFd<> tfd;
    std::cout << tfd << '\n';
  }

  // TimerFdConstructsWithExplicitInputValues
  {
    std::cout << "\n TimerFdConstructsWithExplicitInputValues\n";

    const TimerFd<> tfd {IntervalTimerSpecification{10s, 15s}};
    std::cout << tfd << '\n';

    const TimerFd<> tfd1 {IntervalTimerSpecification{13ms}};
    std::cout << tfd1 << '\n';
  }

  // TimerFdSetTimeWorks
  {
    std::cout << "\n TimerFdSetTimeWorks\n";
    TimerFd<> tfd {IntervalTimerSpecification{10us, 15us}};
    std::cout << tfd << '\n'; // 0 15000 0 10000 0 0 0 0

    tfd.set_time<
      static_cast<std::underlying_type_t<SetTimeFlags>>(
        SetTimeFlags::absolute_time)
      >(IntervalTimerSpecification{5s, 8s});
    std::cout << tfd << '\n'; // 8 0 5 0 0 0 0 0
  }

  // TimerFdGetTimeGetsTimeAfterSetTime
  {
    std::cout << "\n TimerFdGetTimeGetsTimeAfterSetTime\n";
    TimerFd<> tfd {IntervalTimerSpecification{10us, 15us}};
    tfd.get_time();
    std::cout << tfd.current_value() << '\n';
    std::this_thread::sleep_for(2us);
    tfd.get_time();
    std::cout << tfd.current_value() << '\n';

    tfd.set_time<
      static_cast<std::underlying_type_t<SetTimeFlags>>(
        SetTimeFlags::absolute_time)
      >(IntervalTimerSpecification{5s, 8s});
    std::cout << tfd;

    tfd.get_time();
    std::cout << tfd.current_value() << '\n';
    std::this_thread::sleep_for(2s);
    tfd.get_time();
    std::cout << tfd.current_value() << '\n';
  }

  // TimerFdGetsTimesCorrectlyBeforeFirstExpiration
  {
    std::cout << "\n TimerFdGetsTimesCorrectlyBeforeFirstExpiration \n";

    TimerFd<> tfd {IntervalTimerSpecification{10s}};
    std::cout << tfd << '\n';
    tfd.get_time();
    std::cout << tfd.current_value() << '\n';

    for (int i {1}; i <= 5; ++i)
    {
      std::this_thread::sleep_for(2s);
      tfd.get_time();
      std::cout << tfd.current_value() << '\n';
    }
  }

  // TimerFdReadsAndGivesCorrectExpirations
  {
    std::cout << "\n TimerFdReadsAndGivesCorrectExpirations\n";
    TimerFd<> tfd {IntervalTimerSpecification{5s, 5s}};
    std::cout << tfd << '\n';

    for (int delta_t {0}; delta_t < 2; ++delta_t)
    {
      std::cout << " delta_t : " << delta_t << '\n';
      std::this_thread::sleep_for(Seconds{delta_t});
      tfd.read();
      std::cout << "expirations() : " << tfd.expirations() << '\n';
    }

    TimerFd<> tfd1 {IntervalTimerSpecification{250ms, 250ms}};
    std::cout << tfd1 << '\n';

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
    TimerFd<> tfd {IntervalTimerSpecification{50ms, 50ms}};
    std::cout << tfd.get_time();


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
    TimerFd<> tfd {IntervalTimerSpecification{0ms, 250ms}};
    std::cout << tfd.get_time() << '\n';
    std::cout << tfd << '\n';

    for (int delta_t {0}; delta_t < 7; ++delta_t)
    {
      TimerFd<> tfd {IntervalTimerSpecification{0ms, 50ms}};
      std::cout << tfd.get_time() << '\n';
      std::cout << tfd << '\n';

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
