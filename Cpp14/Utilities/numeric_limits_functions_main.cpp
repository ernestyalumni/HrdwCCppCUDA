//------------------------------------------------------------------------------
/// \file numeric_limits_functions_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Main driver file demonstrating numeric limits functions
/// \ref    https://en.cppreference.com/w/cpp/types/numeric_limits
/// \details Member functions of std::numeric_limits class template.
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
///   g++ -std=c++17 numeric_limits_functions_main.cpp -o numeric_limits_functions_main
//------------------------------------------------------------------------------
#include <limits> // std::numeric_limits
#include <iostream>

int main()
{
  //----------------------------------------------------------------------------
  /// \brief epsilon - returns max rounding error of given floating-point type
  /// \ref https://en.cppreference.com/w/cpp/types/numeric_limits/epsilon
  //----------------------------------------------------------------------------
  {
    std::cout << std::numeric_limits<float>::epsilon() << '\n';
    std::cout << std::numeric_limits<double>::epsilon() << '\n';
    std::cout << std::numeric_limits<long double>::epsilon() << '\n';

  }
}
