//------------------------------------------------------------------------------
/// \file InterfaceInheritance.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Interface inheritance as classes only.
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
#include "InterfaceInheritance.h"

#include <iostream>

namespace ClassHierarchy
{

void A::f()
{
  std::cout << "A::f()" << std::endl;
}

A1::A1() = default;

B1::B1() = default;

Implementation1::Implementation1() = default;

Implementation1::Implementation1(const int value):
  value_{value}
{}

Implementation1::~Implementation1() = default;

int Implementation1::value1()
{
  return value_;
}

void Implementation1::set_value1(const int i)
{
  value_ = i;
}

void Implementation1::run1()
{
  std::cout << "\n Implementation1 run1 \n";
}

void B::f()
{  
  std::cout << "B::f()" << std::endl;
}

IoObj* IoDate::clone()
{
  return this;
}

} // namespace ClassHierarchy