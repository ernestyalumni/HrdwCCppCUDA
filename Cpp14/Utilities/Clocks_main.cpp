//------------------------------------------------------------------------------
/// \file Clocks_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  POSIX clock(s) main driver file.
/// \ref      
/// \details 
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
///  g++ -std=c++14 Clocks_main.cpp -o Clocks_main
//------------------------------------------------------------------------------
#include "../Std/TypeTraitsProperties.h"
#include "Chrono.h"
#include "Clocks.h"
#include "casts.h" // get_underlying_value

#include <ctime> // clock_t
#include <iostream>
#include <thread> // std::this_thread
#include <type_traits>

using Std::CompositeTypeTraits;
using Std::PrimaryTypeTraits;
using Utilities::ClockIDs;
using Utilities::ProcessorTimeClock;
using Utilities::TimeSpecification;
using Utilities::get_clock_resolution;
using Utilities::get_clock_time;
using Utilities::set_clock_time;

using namespace Utilities::Literals;

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
    std::cout << " ClockIDs::boot_time : " << 
      static_cast<int>(ClockIDs::boot_time) << '\n'; // 7 
    std::cout << " ClockIDs::real_time_alarm : " << 
      static_cast<int>(ClockIDs::real_time_alarm) << '\n'; // 8
    std::cout << " ClockIDs::boot_time_alarm : " << 
      static_cast<int>(ClockIDs::boot_time_alarm) << '\n'; // 9 
  }

  // TimeSpecificationConstructs
  {
    std::cout << "\n TimeSpecificationConstructs \n";
    const TimeSpecification time_specification {10, 13};
    std::cout << " time_specification : " << time_specification << '\n';
  }

  // TimeSpecificationDefaultConstructsTo0
  {
    std::cout << "\n TimeSpecificationDefaultConstructsTo0 \n";
    const TimeSpecification time_specification;
    std::cout << " time_specification : " << time_specification << '\n';
  }

  // GetClockTimeRetrievesTimesForClocks
  {
    std::cout << "\nGetClockTimeRetrievesTimesForClocks\n";
    TimeSpecification time_specification;
    get_clock_time(time_specification);
    std::cout << " time_specification : " << time_specification << '\n';    

    get_clock_time<ClockIDs::real_time>(time_specification);
    std::cout << " time_specification (real time): " << time_specification <<
      '\n';

    get_clock_time<ClockIDs::process_cpu_time>(time_specification);
    std::cout << " time_specification (CPU time for a process): " <<
      time_specification << '\n';

    get_clock_time<ClockIDs::thread_cpu_time>(time_specification);
    std::cout << " time_specification (CPU time for a thread): " <<
      time_specification << '\n';

    get_clock_time(time_specification);
    std::cout << " time_specification : " << time_specification << '\n';    

    get_clock_time<ClockIDs::real_time>(time_specification);
    std::cout << " time_specification (real time): " << time_specification <<
      '\n';

    get_clock_time<ClockIDs::process_cpu_time>(time_specification);
    std::cout << " time_specification (CPU time for a process): " <<
      time_specification << '\n';

    get_clock_time<ClockIDs::thread_cpu_time>(time_specification);
    std::cout << " time_specification (CPU time for a thread): " <<
      time_specification << '\n';    
  }

  // GetClockResolutionRetrievesResolutionForClocks
  {
    std::cout << "\nGetClockResolutionRetrievesResolutionForClocks\n";
    TimeSpecification time_specification;
    get_clock_resolution(time_specification);
    std::cout << " time_specification : " << time_specification << '\n';    

    get_clock_resolution<ClockIDs::real_time>(time_specification);
    std::cout << " time_specification (real time): " << time_specification <<
      '\n';

    get_clock_resolution<ClockIDs::process_cpu_time>(time_specification);
    std::cout << " time_specification (CPU time for a process): " <<
      time_specification << '\n';

    get_clock_resolution<ClockIDs::thread_cpu_time>(time_specification);
    std::cout << " time_specification (CPU time for a thread): " <<
      time_specification << '\n';
  }

  // clock_tTypeTraitProperties
  {
    std::cout << "\n clock_tTypeTraitProperties \n";

    PrimaryTypeTraits<clock_t> clock_t_primary_type_traits;

    std::cout << clock_t_primary_type_traits << '\n';

    CompositeTypeTraits<clock_t> clock_t_composite_type_traits;

    std::cout << clock_t_composite_type_traits << '\n';

  }

  // ProcessorTimeClockDefaultConstructs
  {
    std::cout << "\n ProcessorTimeClockDefaultConstructs \n";
    ProcessorTimeClock processor_time_clock;
    std::cout << processor_time_clock.last_processor_time_obtained() << '\n';

    std::this_thread::sleep_for(250ms);

    std::cout << processor_time_clock.elapsed_time() << '\n';
  }
}
