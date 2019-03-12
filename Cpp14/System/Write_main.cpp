//------------------------------------------------------------------------------
/// \file Write_main.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief Main driver file for ::write as a C++ functor 
/// \ref http://man7.org/linux/man-pages/man2/write.2.html
/// \details Write to a fd.
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
///  g++ -std=c++17 -I ../ Write.cpp Close.cpp Fork.cpp Open.cpp ../Utilities/Errno.cpp ../Utilities/ErrorHandling.cpp Write_main.cpp -o Write_main
//------------------------------------------------------------------------------
#include "Write.h"

#include <iostream>

int main()
{
  // ssize_tProperties
  {
    std::cout << "\n ssize_tProperties\n";
    std::cout << " sizeof(ssize_t) : " << sizeof(ssize_t) << '\n'; // 8
  }
}