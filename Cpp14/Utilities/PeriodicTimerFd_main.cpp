//------------------------------------------------------------------------------
/// \file PeriodicTimerFd_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  A timerfd as RAII, with expire once timer to count expirations.
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
///  g++ -std=c++14 PeriodicTimerFd_main.cpp -o PeriodicTimerFd_main
//------------------------------------------------------------------------------
#include "Chrono.h"
#include "PeriodicTimerFd.h"

#include <iostream>
#include <thread> // std::this_thread

using Utilities::Microseconds;
using Utilities::Milliseconds;
using Utilities::Nanoseconds;
using Utilities::Seconds;
using Utilities::PeriodicTimerFd;

using namespace Utilities::Literals;


int main()
{

  // PeriodicTimerFdConstructsWithExplicitInputValues
  {
    std::cout << "\n PeriodicTimerFdConstructsWithExplicitInputValues\n";
    PeriodicTimerFd<> ptfd {10s};
    std::cout << ptfd << '\n';
    std::cout << " ptfd.get_time() : " << ptfd.get_time() << '\n';

    PeriodicTimerFd<> ptfd1 {13ms};
    std::cout << ptfd1 << '\n';
    std::cout << " ptfd1.get_time() : " << ptfd1.get_time() << '\n';

  }

  // PeriodicTimerFdReadsAndGivesCorrectExpirations
  {
    PeriodicTimerFd<> ptfd {250ms};
    std::cout << ptfd << '\n';
  
    for (int delta_t {0}; delta_t < 8; ++delta_t)
    {
      std::cout << " delta_t : " << delta_t << '\n';
      std::this_thread::sleep_for(Milliseconds{125 * delta_t});
      ptfd.read_and_block();
      std::cout << "expirations_count() : " << ptfd.expirations_count() <<
        '\n';
    }

    PeriodicTimerFd<> ptfd1 {50ms};
    std::cout << ptfd1 << '\n';

    for (int delta_t {0}; delta_t < 17; ++delta_t)
    {
      std::cout << " delta_t : " << delta_t << '\n';
      std::this_thread::sleep_for(Milliseconds{10 * delta_t});
      ptfd1.read_and_block();
      std::cout << "expirations_count() : " << ptfd1.expirations_count() <<
        '\n';
    }
  }

  // PeriodicTimerFdReadsAndGivesCorrectExpirationsForLessThanAnInterval
  {
    PeriodicTimerFd<> ptfd {250ms};
    std::cout << ptfd << '\n';
  
    for (int delta_t {0}; delta_t < 12; ++delta_t)
    {
      std::cout << " delta_t : " << delta_t << '\n';
      std::this_thread::sleep_for(Milliseconds{125});
      ptfd.read_and_block();
      std::cout << "expirations_count() : " << ptfd.expirations_count() <<
        '\n';
    }


  }

  
}