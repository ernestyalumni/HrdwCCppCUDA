//------------------------------------------------------------------------------
/// \file ErrorHandling_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Main driver file demonstrating ErrorHandling.h
/// \ref
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
///  g++ -std=c++17 Errno.cpp ErrorHandling.cpp ErrorHandling_main.cpp -o ErrorHandling_main
//------------------------------------------------------------------------------
#include "ErrorHandling.h"

#include "Errno.h" // ErrorNumber

#include <iostream>
#include <system_error>

using Utilities::ErrorHandling::ErrorNumber;
using Utilities::ErrorHandling::HandleRead;
using Utilities::ErrorHandling::HandleReturnValue;

int main()
{

  // HandleReturnValueFunctionCallDoesNothingForNonNegativeValues
  {
    std::cout <<
      " \n HandleReturnValueFunctionCallDoesNothingForNonNegativeValues\n";

    HandleReturnValue()(5, "input argument e was not 0 or positive");

    {
      HandleReturnValue{errno}(5,
        "input argument e was not 0 or positive");
    }
  }

  // HandleReturnValueFunctionCallThrowsForNegativeValues
  {
    std::cout << " \n HandleReturnValueFunctionCallThrowsForNegativeValues\n";

    {
      try
      {
        HandleReturnValue()(-3, "input argument e was not 0 or positive");
      }
      catch (const std::system_error& e)
      {
        std::cout << " e.code() : " << e.code() << '\n';
        std::cout << " e.what() : " << e.what() << '\n';

        const ErrorNumber error_number {e.code()};

        std::cout << error_number.error_code() << '\n';
        std::cout << error_number.error_code().message() << '\n';
      }
    }

    {
      try
      {
        HandleReturnValue{errno}(-3, "input argument e was not 0 or positive");
      }
      catch (const std::system_error& e)
      {
        std::cout << " e.code() : " << e.code() << '\n';
        std::cout << " e.what() : " << e.what() << '\n';

        const ErrorNumber error_number {e.code()};

        std::cout << error_number.error_code() << '\n';
        std::cout << error_number.error_code().message() << '\n';
      }
    }
  }

  // intSizeVsssize_tSize
  {
    std::cout << "\n intSizeVsssize_tSize \n";

    std::cout << " sizeof(int) : " << (sizeof(int)) << '\n';
    std::cout << " sizeof(ssize_t) : " << (sizeof(ssize_t)) << '\n';
    std::cout << (sizeof(int) == sizeof(ssize_t)) << '\n';
  }

  // HandleReadThrowsForNegativeNumberOfBytes
  {
    std::cout << "\n HandleReadThrowsForNegativeNumberOfBytes \n";

      try
      {
        const ssize_t number_of_bytes_negative {-3};

        HandleRead()(number_of_bytes_negative);
      }
      catch (const std::system_error& e)
      {
        std::cout << " e.code() : " << e.code() << '\n';
        std::cout << " e.what() : " << e.what() << '\n';
      }
    }

  // HandleReadPrintsEOFMessageWhenNumberOfBytesIsZero
  {
    std::cout << "\n HandleReadThrowsForNegativeNumberOfBytes \n";

    const ssize_t number_of_bytes_zero {0};

    HandleRead()(number_of_bytes_zero);
  }
}
