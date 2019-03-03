//------------------------------------------------------------------------------
/// \file InterfaceInheritance_main.cpp
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
///  g++ -std=c++17 InterfaceInheritance.cpp InterfaceInheritance_main.cpp -o InterfaceInheritance_main
//------------------------------------------------------------------------------
#include "InterfaceInheritance.h"

#include <iostream>

using ClassHierarchy::Interface1;
using ClassHierarchy::Interface2;
using ClassHierarchy::Interface3;
using ClassHierarchy::Implementation1;
using ClassHierarchy::Implementation3;

int main()
{
  //----------------------------------------------------------------------------
  /// \url https://www.bogotobogo.com/cplusplus/dynamic_cast.php
  //----------------------------------------------------------------------------
  {
    Interface3 a;
    Implementation3 b;
    a.f(); // A::f()
    b.f(); // B::f()

    Interface3* pA = &a;
    Implementation3* pB = &b;
    pA->f(); // A::f()
    pB->f(); // B::f()

    pA = &b;
    // pB = &a; // not allowed
    //pB = dynamic_cast<Implementation3*>(&a); // allowed but it returns NULL
    //std::cout << (pB == nullptr) << '\n';

    //--------------------------------------------------------------------------
    /// \url https://www.bogotobogo.com/cplusplus/dynamic_cast.php
    /// \details This successfully dynamically casts pointers of type Interface3
    /// (base class) to a pointer of type Implementation3 (derived class).
    ///
    /// Even though pBaseImplementation3, &a are pointers of type Interface3,
    /// pBaseImplementation3 points to an object of type Implementation3, while
    /// &a refers to an object of type Interface3, which is an incomplete object
    /// of class Implementation3 (derived).
    //--------------------------------------------------------------------------

    Interface3 *pBaseImplementation3 = new Implementation3;

    //Implementation3* pB2 = dynamic_cast<Implementation3*>(&a);
    //std::cout << (pB2 == nullptr) << '\n';

    Implementation3* pB2 = dynamic_cast<Implementation3*>(pBaseImplementation3);

  }

}