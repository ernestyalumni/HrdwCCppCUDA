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
///  g++ -std=c++17 Clocks_main.cpp -o Clocks_main
//------------------------------------------------------------------------------
#include "Clocks.h"

#include <iostream>

using Time::ClockIDs;

int main()
{
  // ClockIDs
  {
    std::cout << "\n ClockIDs \n";
    std::cout << " ClockIDs::real_time : " <<
      static_cast<int>(ClockIDs::real_time) << '\n'; // 0
    std::cout << " ClockIDs::monotonic : " <<
      static_cast<int>(ClockIDs::monotonic) << '\n'; // 1

    std::cout << " ClockIDs::process_cpu_time : " <<
      static_cast<int>(ClockIDs::process_cpu_time) << '\n'; // 2
    std::cout << " ClockIDs::thread_cpu_time : " <<
      static_cast<int>(ClockIDs::thread_cpu_time) << '\n'; // 3

    std::cout << " ClockIDs::monotonic_raw : " <<
      static_cast<int>(ClockIDs::monotonic_raw) << '\n'; // 4

    std::cout << " ClockIDs::real_time_coarse : " <<
      static_cast<int>(ClockIDs::real_time_coarse) << '\n'; // 5
    std::cout << " ClockIDs::monotonic_coarse : " <<
      static_cast<int>(ClockIDs::monotonic_coarse) << '\n'; // 6

    std::cout << " ClockIDs::boot_time : " <<
      static_cast<int>(ClockIDs::boot_time) << '\n'; // 7
    std::cout << " ClockIDs::real_time_alarm : " <<
      static_cast<int>(ClockIDs::real_time_alarm) << '\n'; // 8
    std::cout << " ClockIDs::boot_time_alarm : " <<
      static_cast<int>(ClockIDs::boot_time_alarm) << '\n'; // 9
  }

}
