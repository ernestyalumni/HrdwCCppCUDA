//------------------------------------------------------------------------------
/// \file Tuple2_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  2-Tuple main driver file.
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
///  g++ -std=c++17 Tuple2.cpp Tuple2_main.cpp -o Tuple2_main
//------------------------------------------------------------------------------
#include "Tuple2.h"

#include <cassert>
#include <iostream>
#include <memory>

using Spaces::AffineVector2Implementation;
using Spaces::Tuple2Implementation;

//------------------------------------------------------------------------------
/// \ref pp. 615, Sec. 21.2.1 Stroustrup
/// \details Most application code is written in terms of (pointers to) this way
/// That way, application doesn't have to know about potentially large number of
/// variants of Tuple2Implementation (or Stroustrup's Ival_box class) concept.
//------------------------------------------------------------------------------
void interact(Tuple2Implementation* tuple2_implementation)
{
  //  tuple2_implementation->prompt(); // alert user

  float x {tuple2_implementation->get_x()};

  if (!(tuple2_implementation->is_an_element_of()))
  {
    std::cout << " Put back x : " << x << '\n';
    tuple2_implementation->set_x(x);
  }
  else
  {
    std::cout << " Not possible? " << '\n';
    assert(!(tuple2_implementation->is_an_element_of()) && "Not possible");
  }
}

void some_function()
{

  std::unique_ptr<Tuple2Implementation> p1 {
    std::make_unique<Tuple2Implementation>(0.0, 5.0)};

  interact(p1.get());
}

int main()
{
  // interactWorks
  {
    std::cout << "\n interactWorks \n";

    Tuple2Implementation tuple2_implementation {3., 6.};

    interact(&tuple2_implementation);
  }

  // somefunctionWorks
  {

  }

}

