//------------------------------------------------------------------------------
/// \file Clock_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Interface inheritance for clocks main driver file.
/// \ref
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
///  g++ -std=c++17 -I ../ ../Utilities/Errno.cpp ../Utilities/ErrorHandling.cpp Specifications.cpp Clocks.cpp Clock_main.cpp -o Clock_main
//------------------------------------------------------------------------------
#include "Clock.h"

#include "Clocks.h"
#include "Utilities/Chrono.h"

#include <iostream>

using Time::POSIXMonotonicClock;
using Time::StdSteadyClock;
using Time::TimeSpecification;
using Utilities::Milliseconds;
using Utilities::Seconds;

using namespace Utilities::Literals;

int main()
{
  // StdSteadyClockInSecondsWorks
  {
    std::cout << "\n StdSteadyClockInSecondsWorks \n";

    StdSteadyClock<Seconds> std_steady_clock_sec;

    std_steady_clock_sec.get_current_time();

    std_steady_clock_sec.store_current_time();

    std_steady_clock_sec.get_stored_time();
  }

  // POSIXMonotonicClockWorks
  {
    std::cout << "\n POSIXMonotonicClockWorks \n";

    POSIXMonotonicClock posix_monotonic_clock;

    std::cout << posix_monotonic_clock.get_current_time() << '\n';

    posix_monotonic_clock.store_current_time();

    std::cout << posix_monotonic_clock.get_stored_time() << '\n';
  }

}
