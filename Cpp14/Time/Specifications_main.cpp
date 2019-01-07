//------------------------------------------------------------------------------
/// \file Specifications_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Main driver file for POSIX time specifications.
/// \ref https://linux.die.net/man/3/clock_gettime
/// http://pubs.opengroup.org/onlinepubs/7908799/xsh/time.h.html
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
///  g++ -std=c++17 -I ../ Specifications_main.cpp Specifications.cpp -o Specifications_main
//------------------------------------------------------------------------------
#include "Specifications.h"

#include "Utilities/Chrono.h"

#include <ctime>
#include <iostream>

using Time::TimeSpecification;
using Time::IntervalTimerSpecification;
using Utilities::Milliseconds;
using Utilities::Nanoseconds;

using namespace Utilities::Literals;

int main()
{
  // ConstructsFromADuration
  {
    std::cout << "\n ConstructsFromADuration \n";

    const TimeSpecification time_specification_0 {Milliseconds{0}};

//    const ::timespec time_spec_0 {time_specification_0};
    std::cout << time_specification_0 << '\n';

    const TimeSpecification time_specification_1 {Milliseconds{32250}};

    std::cout << time_specification_1 << '\n';
  }

  // IntervalTimerSpecificationConstructsFromDuration
  {
    std::cout << "\n IntervalTimerSpecificationConstructsFromDuration \n";

    const IntervalTimerSpecification interval_timer_specification {
      Nanoseconds{123456789123}};

    std::cout << interval_timer_specification << '\n';
  }

  // IntervalTimerSpecificationConstructsFromTwoDuration
  {
    std::cout << "\n IntervalTimerSpecificationConstructsFromTwoDuration \n";

    const IntervalTimerSpecification interval_timer_specification {10s, 15s};

    std::cout << interval_timer_specification << '\n';
  }

  // IntervalTimerSpecificationConstructsFromitimerspec
  {
    std::cout << "\n IntervalTimerSpecificationConstructsFromitimerspec\n";

    const IntervalTimerSpecification interval_timer_specification {
      ::itimerspec{10, 13, 42, 69}};

    std::cout << " interval_timer_specification : " <<
      interval_timer_specification << '\n';

    std::cout <<
      ::itimerspec(interval_timer_specification).it_interval.tv_sec << '\n';
    std::cout <<
      ::itimerspec(interval_timer_specification).it_interval.tv_nsec << '\n';
    std::cout <<
      ::itimerspec(interval_timer_specification).it_value.tv_sec << '\n';
    std::cout <<
      ::itimerspec(interval_timer_specification).it_value.tv_nsec << '\n';
  }

}
