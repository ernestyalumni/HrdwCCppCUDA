//------------------------------------------------------------------------------
/// \file StdErrc_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Main driver file for demonstrating scoped enumeration (enum class)
/// std::errc.
/// \ref https://en.cppreference.com/w/cpp/error/errc     
/// \details Scoped enumeration (enum class) std::errc defines values of 
/// portable error conditions corresponding to POSIX error codes.
/// \copyright If you find this code useful, feel free to donate directly via
/// PayPal (username ernestyalumni or email address above); my PayPal profile:
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
///   g++ --std=c++17 StdErrc_main.cpp -o StdErrc_main
//------------------------------------------------------------------------------
#include <cmath> // std::log
#include <cstring> // std::strerror
#include <iostream>
#include <system_error> // std::errc
#include <thread>
#include <type_traits> // std::is_same

#include "Errno.h"

using Utilities::ErrorHandling::Details::ErrorNumbers;
using Utilities::ErrorHandling::ErrorNumber;

int main()
{
  {
    std::cout << " underlying type for std::errc is : " <<
      (std::is_same<int, std::underlying_type_t<std::errc>>::value ?
        " an int." : "not an int.") << '\n';


    std::cout << " address_family_not_supported : " << 
      static_cast<int>(std::errc::address_family_not_supported) << '\n';

    std::cout << std::strerror(
      static_cast<int>(std::errc::address_family_not_supported)) << '\n';

    // EADDRINUSE
    std::cout << " address_in_use : " << 
      static_cast<int>(std::errc::address_in_use) << '\n';

    std::cout << std::strerror(static_cast<int>(std::errc::address_in_use))<<
      '\n';

    // EADDRNOTAVAIL
    std::cout << " address_not_available : " << 
      static_cast<int>(std::errc::address_not_available) << '\n';

    std::cout << std::strerror(
      static_cast<int>(std::errc::address_not_available)) << '\n';

    // EDOM
    std::cout << " argument_out_of_domain : " << 
      static_cast<int>(std::errc::argument_out_of_domain) << '\n';

    std::cout << std::strerror(
      static_cast<int>(std::errc::argument_out_of_domain)) << '\n';
  }

  // ErrorNumbers
  {
    std::cout << " E2BIG : " << static_cast<int>(ErrorNumbers::e2big) << '\n';
    std::cout << " EACCES : " <<
      static_cast<int>(ErrorNumbers::eacces) << '\n';
    std::cout << " EADDRINUSE : " <<
      static_cast<int>(ErrorNumbers::eaddrinuse) << '\n';
    std::cout << " EADDRNOTAVAIL : " <<
      static_cast<int>(ErrorNumbers::eaddrnotavail) << '\n';
    std::cout << " EDOM : " <<
      static_cast<int>(ErrorNumbers::edom) << '\n';
  }

  // \ref https://en.cppreference.com/w/cpp/string/byte/strerror
  {
    double not_a_number = std::log(-1.0);
    ErrorNumber error_number;
    std::cout << "\n error number : " << error_number.error_number() <<
      " error_condition : " <<
      static_cast<int>(error_number.error_condition()) << " " <<
      error_number.as_string() << '\n';

    if (error_number.error_number() == EDOM)
    {
      std::cout << "\n log(-1) failed: " << error_number.as_string() << '\n';
    }
  } 

  // ErrorNumberDefaultConstructs:
  {
    std::cout << " \n ErrorNumberDefaultConstructs : " << '\n';
    std::cout << std::is_default_constructible<ErrorNumber>::value << '\n';

    ErrorNumber error_number;
    ErrorNumber error_number1 {};
  }

  // \ref https://en.cppreference.com/w/cpp/error/errc
  {
    try
    {
      std::thread().detach(); // detaching a not-a-thread
    }
    catch (const std::system_error& e) 
    {
      std::cout << "Caught a system_error\n";

      std::cout << " what : " << e.what() << '\n';
      std::cout << " code value : " << e.code().value() << '\n';

      std::cout << "Caught system_error with code " << e.code() <<
        " meaning " << e.what() << '\n';

      ErrorNumber error_number;

      std::cout << error_number.error_number() << '\n';
      std::cout << error_number.as_string() << '\n';      

    }
  }

}
