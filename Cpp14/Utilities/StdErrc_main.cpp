//------------------------------------------------------------------------------
/// \file StdErrc_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Main driver file for demonstrating scoped enumeration (enum class)
/// std::errc.
/// \ref https://en.cppreference.com/w/cpp/error/errc     
/// \details Scoped enumeration (enum class) std::errc defines values of 
/// portable error conditions corresponding to POSIX error codes.
/// \copyright If you find this code useful, feel free to donate directly
/// (username ernestyalumni or email address above), going directly to: 
///
/// paypal.me/ernestyalumni
/// 
/// which won't go through a 3rd. party such as indiegogo, kickstarter,
/// patreon (which had gotten hacked bad in its history).
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
#include <iostream>
#include <system_error> // std::errc
#include <thread>
#include <type_traits> // std::is_same

int main()
{
  {
    std::cout << " underlying type for std::errc is : " <<
      (std::is_same<int, std::underlying_type_t<std::errc>>::value ?
        " an int." : "not an int.") << '\n';

    std::cout << " address_family_not_supported : " << 
      static_cast<int>(std::errc::address_family_not_supported) << '\n';

  }

  {

  }
}
