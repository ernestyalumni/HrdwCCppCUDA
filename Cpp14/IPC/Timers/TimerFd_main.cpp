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
#include "TimerFd.h"

#include <chrono>
#include <iostream>
#include <thread>

using IPC::Time::ClockIDs;
using IPC::Time::ClockFlags;
using IPC::Time::IntervalTimerSpecification;
using IPC::Time::TimerFd;

int main()
{

  // ClockIDs
  std::cout << "\n ClockIDs \n";  
  std::cout << " ClockIDs::real_time : " << 
    static_cast<int>(ClockIDs::real_time) << '\n'; // 0
  std::cout << " ClockIDs::monotonic : " << 
    static_cast<int>(ClockIDs::monotonic) << '\n'; // 1
  std::cout << " ClockIDs::boot_time : " << 
    static_cast<int>(ClockIDs::boot_time) << '\n'; // 7 
  std::cout << " ClockIDs::real_time_alarm : " << 
    static_cast<int>(ClockIDs::real_time_alarm) << '\n'; // 8
  std::cout << " ClockIDs::boot_time_alarm : " << 
    static_cast<int>(ClockIDs::boot_time_alarm) << '\n'; // 9 


  // ClockFlagsHasAllClockFlagsToChangeBehaviorOftimerfd_create
  {
    std::cout << "\n ClockFlags \n";  
    std::cout << " ClockFlags::non_blocking : " << 
      static_cast<int>(ClockFlags::non_blocking) << '\n'; // 0
    std::cout << " ClockFlags::close_on_execute : " << 
      static_cast<int>(ClockFlags::close_on_execute) << '\n'; // 1
  }

  // IntervalTimerSpecificationConstructsWithExplicitInputValues
  {
    const IntervalTimerSpecification its {1, 2, 3, 4};
    std::cout << its << '\n';
  }

  // TimerFdConstructsWithExplicitInputValues
  {
    const TimerFd<> tfd {
      IntervalTimerSpecification{1, 2, 3, 4},
      IntervalTimerSpecification{5, 6, 7, 8}};
    std::cout << tfd << '\n';
  }

  // TimerFdSetsTimes
  {
    TimerFd<> tfd {IntervalTimerSpecification{15, 0, 10, 0}};
    std::cout << tfd << '\n';
    
    tfd.set_time<>();
    std::cout << tfd << '\n';

    // TimerFdGetsTimes
    {
      std::cout << tfd << '\n';
      IntervalTimerSpecification current_time;
      tfd.get_time(current_time);
      std::cout << current_time << '\n';
      tfd.get_time(current_time);
      std::cout << current_time << '\n';
      tfd.read();
      std::cout << "  number of expirations : " << tfd.expirations() << '\n';
    }

    //TimerFdSetsTimesWithRelativeTime
    {
      TimerFd<> tfd {IntervalTimerSpecification{15, 0, 10, 0}};
      std::cout << tfd << '\n';

      tfd.set_time<0, true>();

      // TimerFdGetsTimes
      {
        std::cout << tfd << '\n';
        IntervalTimerSpecification current_time;
        tfd.get_time(current_time);
        std::cout << current_time << '\n';
        tfd.get_time(current_time);
        std::cout << current_time << '\n';
      }
    // : Resource temporarily unavailable
    //  std::cout << "  number of expirations : " << tfd.read() << '\n';
    }
    //: Resource temporarily unavailable
    //std::cout << "  number of expirations : " << tfd.read() << '\n';
  }

  //MultipleReads
  {
    std::cout << "\n MultipleReads \n";    
    TimerFd<> tfd {IntervalTimerSpecification{10, 0, 5, 0}};
    std::cout << tfd << '\n';
    tfd.set_time<>();
    tfd.read();
    std::cout << "  number of expirations : " << tfd.expirations() << '\n';
  }

  //ReadYieldsCorrectExpirations
  {
    std::cout << "\n ReadYieldsCorrectExpirations\n";
//    std::this_thread::sleep_for

    TimerFd<> tfd {IntervalTimerSpecification{5, 0, 5, 0}};
    std::cout << tfd << '\n';
    for (int tau {1}; tau < 12; ++tau)
    {


    }

  }
  
}
