//------------------------------------------------------------------------------
/// \file Timer_main.cpp
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
///  g++ -std=c++17 -I ../ ../Utilities/Errno.cpp ../Utilities/ErrorHandling.cpp Specifications.cpp Clocks.cpp Timer_main.cpp -o Timer_main
//------------------------------------------------------------------------------
#include "Timer.h"

#include "Specifications.h"
#include "Utilities/Chrono.h"

#include <iostream>

using Time::POSIXMonotonicClockTimer;
using Time::StdSteadyClockTimer;
using Time::TimeSpecification;
using Utilities::Milliseconds;
using Utilities::Nanoseconds;
using Utilities::Seconds;

using namespace Utilities::Literals;

int main()
{
  // StdSteadyClockTimerWorksWithMilliseconds
  {
    std::cout << "\n StdSteadyClockTimerWorksWithMilliseconds \n";

    StdSteadyClockTimer<Milliseconds> std_steady_clock_timer_ms {10ms};

    std::cout << std_steady_clock_timer_ms.expiration_time().count() << '\n';
    // 10
    std::cout << std_steady_clock_timer_ms.elapsed_time().count() << '\n';

    std::cout << std_steady_clock_timer_ms.countdown_time().count() << '\n';

    std::cout << std_steady_clock_timer_ms.is_expired() << '\n';

    std::cout << "\n Print this message \n";

    std::cout << std_steady_clock_timer_ms.expiration_time().count() << '\n';
    // 10
    std::cout << std_steady_clock_timer_ms.elapsed_time().count() << '\n';

    std::cout << std_steady_clock_timer_ms.countdown_time().count() << '\n';

    std::cout << std_steady_clock_timer_ms.is_expired() << '\n';
  }

  // StdSteadyClockTimerWorksWithNanoseconds
  {
    std::cout << "\n StdSteadyClockTimerWorksWithNanoseconds \n";

    StdSteadyClockTimer<Nanoseconds> std_steady_clock_timer_ns {60000ns};

    std::cout << std_steady_clock_timer_ns.expiration_time().count() << '\n';
    // 10
    std::cout << std_steady_clock_timer_ns.elapsed_time().count() << '\n';

    std::cout << std_steady_clock_timer_ns.countdown_time().count() << '\n';

    std::cout << std_steady_clock_timer_ns.is_expired() << '\n';

    std::cout << "\n Print out values \n";

    std::cout << std_steady_clock_timer_ns.expiration_time().count() << '\n';
    // 10
    std::cout << std_steady_clock_timer_ns.elapsed_time().count() << '\n';

    std::cout << std_steady_clock_timer_ns.countdown_time().count() << '\n';

    std::cout << std_steady_clock_timer_ns.is_expired() << '\n';

    std::cout << "\n Print out values again \n";

    std::cout << std_steady_clock_timer_ns.expiration_time().count() << '\n';
    // 10
    std::cout << std_steady_clock_timer_ns.elapsed_time().count() << '\n';

    std::cout << std_steady_clock_timer_ns.countdown_time().count() << '\n';

    std::cout << std_steady_clock_timer_ns.is_expired() << '\n';

  }

  // POSIXMonotonicClockTimerWorks
  {
    std::cout << "\n POSIXMonotonicClockTimerWorks \n";

    POSIXMonotonicClockTimer posix_monotonic_clock_timer {
      TimeSpecification{80000ns}};

    std::cout << posix_monotonic_clock_timer.expiration_time() << '\n';

    std::cout << posix_monotonic_clock_timer.elapsed_time() << '\n';

    std::cout << posix_monotonic_clock_timer.countdown_time() << '\n';

    std::cout << posix_monotonic_clock_timer.is_expired() << '\n';

    std::cout << "\n Print out values \n";

    std::cout << posix_monotonic_clock_timer.expiration_time() << '\n';

    std::cout << posix_monotonic_clock_timer.elapsed_time() << '\n';

    std::cout << posix_monotonic_clock_timer.countdown_time() << '\n';

    std::cout << posix_monotonic_clock_timer.is_expired() << '\n';

    std::cout << "\n Print out values again \n";

    std::cout << posix_monotonic_clock_timer.expiration_time() << '\n';

    std::cout << posix_monotonic_clock_timer.elapsed_time() << '\n';

    std::cout << posix_monotonic_clock_timer.countdown_time() << '\n';

    std::cout << posix_monotonic_clock_timer.is_expired() << '\n';
  }
}
