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
using ClassHierarchy::A;
using ClassHierarchy::A1;
using ClassHierarchy::Implementation1;
using ClassHierarchy::B;
using ClassHierarchy::B1;
using ClassHierarchy::IoDate;
using ClassHierarchy::IoObj;

int main()
{
  //----------------------------------------------------------------------------
  /// \url https://www.bogotobogo.com/cplusplus/dynamic_cast.php
  //----------------------------------------------------------------------------
  {
    A a;
    B b;
    a.f(); // A::f()
    b.f(); // B::f()

    A* pA = &a;
    B* pB = &b;
    pA->f(); // A::f()
    pB->f(); // B::f()

    pA = &b; // B* -> A* 
    // pB = &a; // not allowed
    //pB = dynamic_cast<B*>(&a); // allowed but it returns NULL
    //std::cout << (pB == nullptr) << '\n';

    //--------------------------------------------------------------------------
    /// \url https://www.bogotobogo.com/cplusplus/dynamic_cast.php
    /// \details This successfully dynamically casts pointers of type A
    /// (base class) to a pointer of type B (derived class).
    ///
    /// Even though pBaseB, &a are pointers of type A,
    /// pBaseB points to an object of type B, while
    /// &a refers to an object of type A, which is an incomplete object
    /// of class B (derived).
    //--------------------------------------------------------------------------

    A* pBaseB = new B; 

    //B* pB2 = dynamic_cast<B*>(&a);
    //std::cout << (pB2 == nullptr) << '\n';

    B* pB2 = dynamic_cast<B*>(pBaseB);

    //--------------------------------------------------------------------------
    /// \url https://stackoverflow.com/questions/2362097/why-is-the-size-of-an-empty-class-in-c-not-zero
    /// \details The standard does not allow objects (and classes thereof) of
    /// size 0, since that would make it possible for two distinct objects to
    /// have the same memory address. That's why even empty classes must have a
    /// size of (at least) 1.
    //--------------------------------------------------------------------------
    std::cout << " sizeof(A1) : " << sizeof(A1) << '\n';

    B1* b1;
    A* b1_as_A = b1; // OK, B1 -> A
    A* b1_as_A_2 = dynamic_cast<A*>(b1); // OK, B1 -> A via dynamic_cast

    //--------------------------------------------------------------------------
    /// \details error: is an inaccessible base
    /// \ref pp. 643 Stroustrup, dynamic_cast doesn't allow accidental violation
    /// of protection of private and protected base classes. Since a
    /// dynamic_cast used as an upcast is exactly like a simple assignment, it
    /// implies no overhead and is sensitive to its lexical context.
    //--------------------------------------------------------------------------
    //A1* b1_as_A1 = b1; // B1 -> A1, but A1 protected
    //A1* b1_as_A1 = dynamic_cast<B1*>(b1); 

    //--------------------------------------------------------------------------
    /// \details error: is an inaccessible base
    /// \ref pp. 645 Stroustrup, 
    //--------------------------------------------------------------------------

    IoObj* pio_obj;
    // Segmentation fault
    //A* pio_obj_as_pA = dynamic_cast<A*>(pio_obj);

    // Segmentation fault
    // IoDate* pio_obj_as_pIoDate = dynamic_cast<IoDate*>(pio_obj);

    //--------------------------------------------------------------------------
    /// \details dynamic_cast to void* can be used to determine address of the
    /// beginning of an object of polymorphic type.
    /// \ref pp. 645 Stroustrup, 
    //--------------------------------------------------------------------------
    void* address_of_pBaseB = dynamic_cast<void*>(pBaseB);
    std::cout << " address of pBaseB : " << address_of_pBaseB << ' ' <<
      &address_of_pBaseB << '\n';

    void* address_of_pB2 = dynamic_cast<void*>(pB2);
    std::cout << " address of pB2 : " << address_of_pB2 << ' ' <<
      &address_of_pB2 << '\n';

    std::cout << " address_of_pBaseB should be equal to address_of_pB2 : " <<
      (address_of_pBaseB == address_of_pB2) << '\n';

  }
}