//------------------------------------------------------------------------------
/// \file Chrono_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Date and time utilities main driver file.
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
///  g++ -std=c++17 Chrono_main.cpp -o Chrono_main
//------------------------------------------------------------------------------
#include "Chrono.h"

#include "../Std/TypeTraitsProperties.h"

#include <iostream>
#include <thread>
#include <type_traits>

using Std::CompositeTypeTraits;
using Std::PrimaryTypeTraits;
using Std::TypeProperties;
using Utilities::ElapsedTime;
using Utilities::HighResolutionClock;
using Utilities::Microseconds;
using Utilities::Milliseconds;
using Utilities::Nanoseconds;
using Utilities::Seconds;
using Utilities::SteadyClock;
using Utilities::SteadyTimePoint;
using Utilities::SystemClock;
using Utilities::SystemTime;
using Utilities::duration_cast;

using namespace Utilities::Literals;

int main()
{
  // ClocksGiveTimeNow
  {
    // \ref https://stackoverflow.com/questions/45000569/recursion-with-generic-lambda-functions-in-c14
    // https://stackoverflow.com/questions/18085331/recursive-lambda-functions-in-c14/18085333#18085333
    // \details auto&& means "I don't mind if this is temporary or whatever,
    // just don't make a copy"
    constexpr auto fibonacci = [&](auto&& self, int n) -> int
    {
      if (n < 2)
      {
        return n; // return type is deduced here
      }
      else
      {
        return self(self, n - 1) + self(self, n - 2);
      }
    };

    std::cout << "\n ClocksGiveTimeNow \n";
    const auto start = SystemClock::now();
    std::cout << "f(42) = " << fibonacci(fibonacci, 42) << '\n';
    const auto end = SystemClock::now();

    const Milliseconds elapsed_ms {duration_cast<Milliseconds>(end - start)};
    const auto end_time = SystemClock::to_time_t(end);

    std::cout << "finished computation at " << end_time << '\n';
    std::cout << "elapsed time (ms): " << elapsed_ms.count() << '\n';
  }

  // DurationHelperTypesConvert
  {
    std::cout << "\n DurationHelperTypesConvert \n";
    const Seconds sec {1};
    std::cout << " sec.count() : " << sec.count() << '\n'; // 1
    const Microseconds usec {sec};
    std::cout << " usec.count() : " << usec.count() << '\n'; // 1000000

    // Integer scale conversion with precision loss: requires a cast
    std::cout << duration_cast<Microseconds>(Nanoseconds{1}).count() << '\n';
    // 0
  }

  // DurationCastMeasuresExecutionTimeOfAFunction
  // \ref https://en.cppreference.com/w/cpp/chrono/duration/duration_cast
  {
    std::cout << "\n DurationCastMeasuresExecutionTimeOfAFunction \n";
    auto f = [](){ std::this_thread::sleep_for(std::chrono::seconds(1)); };
    const auto t1 = HighResolutionClock::now();
    f();
    const auto t2 = HighResolutionClock::now();

    // Integral duration; requires duration_cast
    const auto int_ms = duration_cast<Milliseconds>(t2 - t1); // 1000
    std::cout << " int_ms.count() : " << int_ms.count() << '\n';
  }

  // DurationCastRoundsToZero
  {
    std::cout << "\n DurationCastRoundsToZero \n";

    {
      const Microseconds t_0_us {123456789};
      const Seconds t_0_s {duration_cast<Seconds>(t_0_us)};

      std::cout << " t_0_ms.count() : " << t_0_us.count() << '\n';
      std::cout << " t_0_s.count() : " << t_0_s.count() << '\n';
    }

    {
      const Milliseconds t_0_ms {-123456};
      const Seconds t_0_s {duration_cast<Seconds>(t_0_ms)};

      std::cout << " t_0_ms.count() : " << t_0_ms.count() << '\n';
      std::cout << " t_0_s.count() : " << t_0_s.count() << '\n';
    }
  }

  // ElapsedTimeConstructs
  {
    const ElapsedTime<Milliseconds, SteadyClock> elapsed_time;
    const ElapsedTime<Milliseconds, SteadyClock> elapsed_time_1;
  }

  // ElapsedTimeCountsMicroseconds
  {
    std::cout << "\n ElapsedTimeCountsMicroseconds \n";
    ElapsedTime<Microseconds> elapsed_time;
    std::cout << "Hello World\n";
    const auto tau = elapsed_time();
    std::cout << "Printing took : " << tau.count() << " us." << '\n'; // 2
  }

  // ChronoLiteralsConstructDurationHelperTypes
  {
    std::cout << "\n ChronoLiteralsConstructDurationHelperTypes \n";
    const Nanoseconds nanosecs {42ns};
    std::cout << " nanosecs.count() : " << nanosecs.count() << '\n';
  }

  // DurationHelperTypesHasTypeProperties
  {
    std::cout << "\nDurationHelperTypesHasTypePropertes\n";

    std::cout << " is_integral : " << std::is_integral<long>::value <<
      '\n'; // 1
    std::cout << " is_integral : " << std::is_integral<const long>::value <<
      '\n'; // 1

    std::cout << " is_integral : " << std::is_integral<Nanoseconds>::value <<
      '\n'; // 0
    std::cout << " is_integral : " << std::is_integral<Microseconds>::value <<
      '\n'; // 0
    std::cout << " is_integral : " << std::is_integral<Milliseconds>::value <<
      '\n'; // 0
    std::cout << " is_integral : " << std::is_integral<Seconds>::value <<
      '\n'; // 0

    std::cout << " is_floating_point : " <<
      std::is_floating_point<long>::value << '\n'; // 0
    std::cout << " is_floating_point : " <<
      std::is_floating_point<const long>::value << '\n'; // 0

    std::cout << " is_floating_point : " <<
      std::is_floating_point<Nanoseconds>::value << '\n'; // 0
    std::cout << " is_floating_point : " <<
      std::is_floating_point<Microseconds>::value << '\n'; // 0
    std::cout << " is_floating_point : " <<
      std::is_floating_point<Milliseconds>::value << '\n'; // 0
    std::cout << " is_floating_point : " <<
      std::is_floating_point<Seconds>::value << '\n'; // 0

    std::cout << " is_class : " << std::is_class<long>::value << '\n'; // 0
    std::cout << " is_class : " << std::is_class<const long>::value << '\n'; // 0

    std::cout << " is_class : " << std::is_class<Nanoseconds>::value << '\n';
      // 1
    std::cout << " is_class : " << std::is_class<Microseconds>::value << '\n';
      // 1
    std::cout << " is_class : " << std::is_class<Milliseconds>::value << '\n';
      // 1
    std::cout << " is_class : " << std::is_class<Seconds>::value << '\n'; // 0

    std::cout << " is_compound : " << std::is_compound<long>::value << '\n';
      // 0
    std::cout << " is_compound : " << std::is_compound<const long>::value <<
      '\n'; // 0

    std::cout << " is_compound : " << std::is_compound<Nanoseconds>::value <<
      '\n'; // 1
    std::cout << " is_compound : " << std::is_compound<Microseconds>::value <<
      '\n'; // 1
    std::cout << " is_compound : " << std::is_compound<Milliseconds>::value <<
      '\n'; // 1
    std::cout << " is_compound : " << std::is_compound<Seconds>::value <<
      '\n'; // 1
  }

  // PrimaryTypeTraits
  {
    std::cout << "\n PrimaryTypeTraits \n";

    std::cout << PrimaryTypeTraits<Nanoseconds>{} << '\n';
  }

  // CompositeTypeTraits
  {
    std::cout << "\n CompositeTypeTraits \n";

    std::cout << CompositeTypeTraits<Nanoseconds>{} << '\n';
  }

  // TypeProperties
  {
    std::cout << "\n TypeProperties \n";

    std::cout << TypeProperties<Nanoseconds>{} << '\n';
  }

  // SteadyClockGetsTime
  {
    std::cout << "\n SteadyClockGetsTime\n";

    SteadyClock::now();
  }

}
