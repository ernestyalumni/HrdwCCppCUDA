//------------------------------------------------------------------------------
/// \file TimerFd_check_read.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  A timerfd as RAII; checking read 
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
///  g++ -std=c++14 -lpthread TimerFd_main.cpp -o TimerFd_main
//------------------------------------------------------------------------------
#include "Chrono.h"
#include "Clocks.h" // ClockIDs
#include "TimerFd.h" // ClockFlags

#include <iostream>
#include <thread> // std::this_thread

// \ref http://man7.org/linux/man-pages/man2/fcntl.2.html
#include <unistd.h>
#include <fcntl.h> // fcntl


using Utilities::ClockFlags;
using Utilities::ClockIDs;
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
  std::cout << "\n TimerFdReadsAndGivesCorrectExpirations\n";
  {
    Seconds test_period {4s};
    TimerFd<ClockIDs::monotonic, ClockFlags::non_blocking> tfd {
      test_period, test_period};
//    std::cout << " tfd : " << tfd << '\n';
    std::cout << " expirations() : " << tfd.expirations() << '\n';


    Seconds test_seconds {1};
//    std::cout << " test_seconds < tfd period : " << (test_seconds < 4s) << '\n';

    tfd.set_time<SetTimeFlags::default_value, true>();

    std::this_thread::sleep_for(test_seconds);

    tfd.read();

    std::cout << " expirations() : " << tfd.expirations() << '\n';

    for (int delta_t {0}; delta_t < 10; ++delta_t)
    {
      std::cout << " delta_t : " << delta_t << '\n';
      test_seconds = Seconds{delta_t};
      std::this_thread::sleep_for(test_seconds);      
      tfd.read();
      std::cout << " expirations() : " << tfd.expirations() << '\n';

    }

    /*
    auto wait_and_read =
      [&tfd](int i)
      {
        std::cout << " i : " << i << '\n';
        std::this_thread::sleep_for(Seconds{i});
        tfd.read();
        std::cout << " expirations() : " << tfd.expirations() << '\n';
      };

    std::thread th(wait_and_read, 0);
    th.join();
  */
  }




}
