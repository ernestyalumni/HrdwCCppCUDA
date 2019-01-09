//------------------------------------------------------------------------------
/// \file POSIXElapsedTime_main.cpp
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
///  g++ -std=c++17 -I ../ ../Utilities/Errno.cpp ../Utilities/ErrorHandling.cpp Specifications.cpp Clocks.cpp POSIXElapsedTime_main.cpp -o POSIXElapsedTime_main
//------------------------------------------------------------------------------
#include "POSIXElapsedTime.h"

#include "Clocks.h" // GetClockTime

#include <iostream>

using Time::GetClockTime;
using Time::POSIXElapsedTime;

int main()
{
  // POSIXElapsedTimeConstructs
  {
    std::cout << " \n POSIXElapsedTimeConstructs \n";

    POSIXElapsedTime posix_elapsed_time;
  }

  // POSIXElapsedTimeStartGetsTime
  {
    std::cout << "\n POSIXElapsedTimeStartGetsTime \n";

    GetClockTime<> get_clock_time;

    POSIXElapsedTime posix_elapsed_time;
    posix_elapsed_time.start();
    std::cout << " posix_elapsed_time.t_0() : " <<
      posix_elapsed_time.t_0() << '\n';

    get_clock_time();
    std::cout << " get_clock_time : " << get_clock_time.time_specification() <<
      '\n';
  }

  // POSIXElapsedTimeCanDoSubtraction
  {
    std::cout << "\n POSIXElapsedTimeCanDoSubtraction \n";

    GetClockTime<> get_clock_time;

    POSIXElapsedTime posix_elapsed_time;
    posix_elapsed_time.start();

    std::cout << " posix_elapsed_time.t_0() : " <<
      posix_elapsed_time.t_0() << '\n';

    get_clock_time();
    std::cout << " get_clock_time : " << get_clock_time.time_specification() <<
      '\n';

    std::cout << posix_elapsed_time() << '\n';

  }

}
