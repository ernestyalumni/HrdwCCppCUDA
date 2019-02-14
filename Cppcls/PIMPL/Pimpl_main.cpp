//------------------------------------------------------------------------------
/// \file Pimpl_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Main driver file.
/// \url https://en.cppreference.com/w/cpp/language/pimpl
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
///
/// \details Every bit in the value representation of the returned To object is
/// equal to the corresponding bit in the object representation of from. The
/// values of padding bits in returned To object are unspecified.
//------------------------------------------------------------------------------
/// COMPILATION TIPS:
///  g++ -std=c++17 Pimpl.cpp Pimpl_main.cpp -o Pimpl_main
//------------------------------------------------------------------------------
#include "Pimpl.h"

using Idioms::Pimpl::Examples::ClassicParser;
using Idioms::Pimpl::Examples::Widget;

int main()
{

  {
    Widget widget {7};
    const Widget widget2 {8};

    widget.draw();
    widget2.draw();
  }

  {
    ClassicParser classic_parser {"abcdef"};
    classic_parser.parse("ghijkl");
  }

}
