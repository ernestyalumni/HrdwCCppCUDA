//------------------------------------------------------------------------------
/// \file Clocks_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  POSIX clock(s) main driver file.
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
///  g++ -std=c++17 -I ../ ../Utilities/Errno.cpp ../Utilities/ErrorHandling.cpp Specifications.cpp Clocks.cpp Clocks_main.cpp -o Clocks_main
//------------------------------------------------------------------------------
#include "Clocks.h"

#include <ctime>
#include <iostream>

using Time::ClockIds;
using Time::carry_nanoseconds_to_seconds;

int main()
{
  // ClockIds
  {
    std::cout << "\n ClockIds \n";
    std::cout << " ClockIds::real_time : " <<
      static_cast<int>(ClockIds::real_time) << '\n'; // 0
    std::cout << " ClockIds::monotonic : " <<
      static_cast<int>(ClockIds::monotonic) << '\n'; // 1

    std::cout << " ClockIds::process_cpu_time : " <<
      static_cast<int>(ClockIds::process_cpu_time) << '\n'; // 2
    std::cout << " ClockIds::thread_cpu_time : " <<
      static_cast<int>(ClockIds::thread_cpu_time) << '\n'; // 3

    std::cout << " ClockIds::monotonic_raw : " <<
      static_cast<int>(ClockIds::monotonic_raw) << '\n'; // 4

    std::cout << " ClockIds::real_time_coarse : " <<
      static_cast<int>(ClockIds::real_time_coarse) << '\n'; // 5
    std::cout << " ClockIds::monotonic_coarse : " <<
      static_cast<int>(ClockIds::monotonic_coarse) << '\n'; // 6

    std::cout << " ClockIds::boot_time : " <<
      static_cast<int>(ClockIds::boot_time) << '\n'; // 7
    std::cout << " ClockIds::real_time_alarm : " <<
      static_cast<int>(ClockIds::real_time_alarm) << '\n'; // 8
    std::cout << " ClockIds::boot_time_alarm : " <<
      static_cast<int>(ClockIds::boot_time_alarm) << '\n'; // 9
  }

  // CarryNanosecondsToSecondsWorks
  {
    std::cout << "\n CarryNanosecondsToSecondsWorks \n";

    ::timespec timespec {0, 1000000000};
    ::timespec result_timespec {carry_nanoseconds_to_seconds(timespec)};
    std::cout << " result_timespec.tv_sec : " << result_timespec.tv_sec <<
      " result_timespec.tv_nsec : " << result_timespec.tv_nsec << '\n'; // 1 0

    timespec = ::timespec{1, 2000000001};
    result_timespec = carry_nanoseconds_to_seconds(timespec);
    std::cout << " result_timespec.tv_sec : " << result_timespec.tv_sec <<
      " result_timespec.tv_nsec : " << result_timespec.tv_nsec << '\n'; // 3 1

    timespec = ::timespec{1, -1000000000};
    result_timespec = carry_nanoseconds_to_seconds(timespec);
    std::cout << " result_timespec.tv_sec : " << result_timespec.tv_sec <<
      " result_timespec.tv_nsec : " << result_timespec.tv_nsec << '\n'; // 0 0

    timespec = ::timespec{-1, -1000000000};
    result_timespec = carry_nanoseconds_to_seconds(timespec);
    std::cout << " result_timespec.tv_sec : " << result_timespec.tv_sec <<
      " result_timespec.tv_nsec : " << result_timespec.tv_nsec << '\n'; // -2 0

    timespec = ::timespec{1, -999999999};
    result_timespec = carry_nanoseconds_to_seconds(timespec);
    std::cout << " result_timespec.tv_sec : " << result_timespec.tv_sec <<
      " result_timespec.tv_nsec : " << result_timespec.tv_nsec << '\n'; // 0 1

    timespec = ::timespec{-1, -1};
    result_timespec = carry_nanoseconds_to_seconds(timespec);
    std::cout << " result_timespec.tv_sec : " << result_timespec.tv_sec <<
      " result_timespec.tv_nsec : " << result_timespec.tv_nsec << '\n';
      // -2 999999999

    timespec = ::timespec{5, -2444444444};
    result_timespec = carry_nanoseconds_to_seconds(timespec);
    std::cout << " result_timespec.tv_sec : " << result_timespec.tv_sec <<
      " result_timespec.tv_nsec : " << result_timespec.tv_nsec << '\n';
    // 2 555555556
  }


}
