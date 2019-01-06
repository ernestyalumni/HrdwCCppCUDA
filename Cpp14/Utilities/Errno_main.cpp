//------------------------------------------------------------------------------
/// \file Errno_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Main driver file for Errno.h.
/// \ref https://en.cppreference.com/w/cpp/error/errno
/// \details C++ Functors
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
///  g++ -std=c++17 Errno.cpp Errno_main.cpp -o Errno_main
//------------------------------------------------------------------------------
#include "Errno.h"

#include <cmath>
#include <iostream>
#include <system_error> // std::errc
#include <thread>

using Utilities::ErrorHandling::ErrorNumber;

int main()
{
  // ErrorNumberDefaultConstructs
  {
    std::cout << " \n ErrorNumberDefaultConstructs \n";

    ErrorNumber error_number;

    std::cout << " error_number.as_string() : " << error_number.as_string() <<
      '\n';

    std::cout << " error_number.error_number() : " <<
      error_number.error_number() << '\n';

    // Checks if value is non-zero
    /// \ref https://en.cppreference.com/w/cpp/error/error_condition
    std::cout << " bool(error_number.error_condition()) : " <<
      bool(error_number.error_condition()) << '\n';

    std::cout << error_number.error_condition().message() << '\n';

    std::cout << error_number.error_condition().value() << '\n';

    std::cout << error_number.error_code() << '\n';

    std::cout << error_number.error_code().value() << '\n';

    // DefaulterrnoIntoErrorCode
    {
      std::cout << "\n DefaulterrnoIntoErrorCode \n";
      std::cout << " errno : " << errno << '\n';
      const std::errc error_constant {static_cast<std::errc>(errno)};
      std::cout << (std::is_error_code_enum<std::errc>::value) << "\n";
      std::cout << (std::is_error_condition_enum<std::errc>::value) << "\n";
    }
  }

  // ErrorNumberDetectsErrnoUponConstruction
  {
    std::cout << "\n ErrorNumberDetectsErrnoUponConstruction \n";

    /// \ref https://en.cppreference.com/w/cpp/string/byte/strerror
    double not_a_number = std::log(-1.0);

    ErrorNumber error_number;

    std::cout << " error_number.as_string() : " << error_number.as_string() <<
      '\n';

    std::cout << " error_number.error_number() : " <<
      error_number.error_number() << '\n';

    std::cout << " bool(error_number.error_condition()) : " <<
      bool(error_number.error_condition()) << '\n';

    std::cout << error_number.error_condition().message() << '\n';

    std::cout << error_number.error_condition().value() << '\n';

    std::cout << error_number.error_code() << '\n';

    std::cout << error_number.error_code().value() << '\n';
  }


  /// \ref https://en.cppreference.com/w/cpp/error/errc

  // ErrnoDetectedForAnyErrorNumberConstruction
  {
    std::cout << "\n ErrnoDetectedForAnyErrorNumberConstruction \n";

    try
    {
      std::thread().detach(); // detaching a not-a-thread
    }
    catch (const std::system_error& e)
    {
      std::cout << "Caught a system_error\n";

      ErrorNumber error_number;

      std::cout << " error_number.as_string() : " << error_number.as_string() <<
        '\n';

      std::cout << " error_number.error_number() : " <<
        error_number.error_number() << '\n';

      std::cout << " bool(error_number.error_condition()) : " <<
        bool(error_number.error_condition()) << '\n';

      std::cout << error_number.error_condition().message() << '\n';
      std::cout << error_number.error_condition().value() << '\n';
      std::cout << error_number.error_code() << '\n';
      std::cout << error_number.error_code().value() << '\n';

      {
        ErrorNumber error_number;

        std::cout << " error_number.as_string() : " << error_number.as_string() <<
          '\n';

        std::cout << " error_number.error_number() : " <<
          error_number.error_number() << '\n';

        std::cout << " bool(error_number.error_condition()) : " <<
          bool(error_number.error_condition()) << '\n';

        std::cout << error_number.error_condition().message() << '\n';
        std::cout << error_number.error_condition().value() << '\n';
        std::cout << error_number.error_code() << '\n';
        std::cout << error_number.error_code().value() << '\n';
      }

      ErrorNumber error_number1;

      std::cout << " error_number1.as_string() : " << error_number1.as_string() <<
        '\n';

      std::cout << " error_number1.error_number() : " <<
        error_number1.error_number() << '\n';

      std::cout << " bool(error_number1.error_condition()) : " <<
        bool(error_number1.error_condition()) << '\n';

      std::cout << error_number1.error_condition().message() << '\n';
      std::cout << error_number1.error_condition().value() << '\n';
      std::cout << error_number1.error_code() << '\n';
      std::cout << error_number1.error_code().value() << '\n';

      ErrorNumber error_number2 {e.code().value(), e.code().category()};

      std::cout << " error_number2.as_string() : " << error_number2.as_string() <<
        '\n';

      std::cout << " error_number2.error_number() : " <<
        error_number2.error_number() << '\n';

      std::cout << " bool(error_number2.error_condition()) : " <<
        bool(error_number2.error_condition()) << '\n';

      std::cout << error_number2.error_condition().message() << '\n';
      std::cout << error_number2.error_condition().value() << '\n';
      std::cout << error_number2.error_code() << '\n';
      std::cout << error_number2.error_code().value() << '\n';
    }

  }


  // ErrorNumberConstructsFromStdErrorCode
  {
    std::cout << "\n ErrorNumberConstructsFromStdErrorCode \n";

    try
    {
      std::thread().detach(); // detaching a not-a-thread
    }
    catch (const std::system_error& e)
    {
      std::cout << "Caught a system_error\n";

      ErrorNumber error_number {e.code()};

      std::cout << " error_number.as_string() : " << error_number.as_string() <<
        '\n';

      std::cout << " error_number.error_number() : " <<
        error_number.error_number() << '\n';

      std::cout << " bool(error_number.error_condition()) : " <<
        bool(error_number.error_condition()) << '\n';

      std::cout << error_number.error_condition().message() << '\n';
      std::cout << error_number.error_condition().value() << '\n';
      std::cout << error_number.error_code() << '\n';
      std::cout << error_number.error_code().value() << '\n';
    }
  }
}
