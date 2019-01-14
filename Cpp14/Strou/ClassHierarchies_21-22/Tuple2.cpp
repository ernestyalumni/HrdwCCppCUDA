//------------------------------------------------------------------------------
/// \file Tuple2.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  2-Tuple source implementation file.
/// \ref Ch. 21 Class Hierarchies, 21.2.Design of Class Hierarchies
///   The C++ Programming Language, 4th Ed., Stroustrup;
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
///  g++ -std=c++17 Tuple2_main.cpp -o Tuple2_main
//------------------------------------------------------------------------------
#include "Tuple2.h"

namespace Spaces
{

Tuple2Implementation::Tuple2Implementation():
  x_{0},
  y_{0}
{}

Tuple2Implementation::Tuple2Implementation(const float x, const float y):
  x_{x},
  y_{y}
{}

AffineVector2Implementation::AffineVector2Implementation(const float x,
  const float y):
  Tuple2Implementation{x, y}
{}

float AffineVector2Implementation::get_x()
{
  return x_;
}

float AffineVector2Implementation::get_y()
{
  return y_;
}


} // namespace Spaces
