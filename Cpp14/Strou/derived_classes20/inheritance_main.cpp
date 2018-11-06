//------------------------------------------------------------------------------
/// \file inheritance_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Main driver function for demonstrating inheritance in C++.  
/// \ref Ch. 20 Derived classes; Bjarne Stroustrup, 
///   The C++ Programming Language, 4th Ed., Stroustrup; Ch.23
/// https://en.cppreference.com/book/intro/inheritance
/// \details 
/// \copyright If you find this code useful, feel free to donate directly and
/// easily at this direct PayPal link: 
///
/// https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
/// 
/// which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.
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
///  g++ -std=c++14 inheritance_main.cpp -o inheritance_main
//------------------------------------------------------------------------------
#include "inheritance.h"

#include <iostream>
#include <type_traits>

using Inheritance::A1;
using Inheritance::A;
using Inheritance::AAbstract;
using Inheritance::B1;
using Inheritance::B;
using Inheritance::BA1;
using Inheritance::BA2;
using Inheritance::Base;
using Inheritance::Employee;
using Inheritance::ImaginaryNumber;
using Inheritance::Manager;
using Inheritance::Number;
using Inheritance::print_list;

int main()
{
  // CannotDeclareVariableOfAbstractType.
  {
    std::cout << "\n CannotDeclareVariableOfAbstractType \n";
    // error: cannot declare variable a to be of abstract type
    // Number a;
    // note: because the following virtual functions are pure within AAbstract
    // AAbstract a;
  }

  // AbstractClassesHaveTypeProperties
  {
    std::cout << "\n AbstractClassesHasTypeProperties \n";

    std::cout << " is_class : " << std::is_class<Number>::value << '\n'; // 1
    std::cout << " is_class : " << std::is_class<AAbstract>::value <<
      '\n'; // 1

    // compound type (array, function, object ptr, function ptr, member object
    // ptr, member function ptr, reference, class, union, or enum)
    std::cout << " is_compound : " << std::is_compound<Number>::value << '\n';
      // 1
    std::cout << " is_compound : " << std::is_compound<AAbstract>::value <<
      '\n'; // 1

    // checks if trivially copyable, and has 1 or more default ctors all
    // trivial or deleted, and at least 1 which isn't deleted
    std::cout << " is_trivial : " << std::is_trivial<Number>::value <<
      '\n'; // 0
    std::cout << " is_trivial : " << std::is_trivial<AAbstract>::value <<
      '\n'; // 0

    // checks if copy, move ctors, copy, move assignment trivial or deleted, at
    // least 1 copy, move ctor, copy, move assignment operator is non-deleted,
    // trivial, non-deleted dtor, implies no virtual.
    std::cout << " is_trivially_copyable : " <<
      std::is_trivially_copyable<Number>::value << '\n'; // 0
    std::cout << " is_trivially_copyable : " <<
      std::is_trivially_copyable<AAbstract>::value << '\n'; // 0

    // checks if scalar type, standard layout type for communicating with code
    // in other langauges; requires all non-static data members have same
    // access control, no virtual functions or virtual base classes, no
    // non-static data members of reference type, all non-static data members
    // and base classes are themselves standard layout types
    std::cout << " is_standard_layout : " <<
      std::is_standard_layout<Number>::value << '\n'; // 0
    std::cout << " is_standard_layout : " <<
      std::is_standard_layout<AAbstract>::value << '\n'; // 0

    // checks if type is plain-old data (POD) type
    std::cout << " is_pod : " << std::is_pod<Number>::value << '\n'; // 0
    std::cout << " is_pod : " << std::is_pod<AAbstract>::value << '\n'; // 0

    // checks if type is class (but not union) and has no data
    std::cout << " is_empty : " << std::is_empty<Number>::value << '\n'; // 0
    std::cout << " is_empty : " << std::is_empty<AAbstract>::value << '\n'; // 0

    // That is non-union class that declares or inherits at least 1 virtual function
    std::cout << " is_polymorphic : " <<
      std::is_polymorphic<Number>::value << '\n'; // 1
    std::cout << " is_polymorphic : " <<
      std::is_polymorphic<AAbstract>::value << '\n'; // 1

    std::cout << " is_abstract : " << std::is_abstract<Number>::value << '\n';
      // 1
    std::cout << " is_abstract : " << std::is_abstract<AAbstract>::value <<
      '\n'; // 1

    std::cout << " is_final : " << std::is_final<Number>::value << '\n'; // 0
    std::cout << " is_final : " << std::is_final<AAbstract>::value << '\n';
      // 0
  }

  // DerivedClassesDefaultConstruct
  {
    std::cout << "\n DerivedClassesDefaultConstruct\n";
    const ImaginaryNumber z;
    std::cout << z.isImaginaryNumber() << ' ' << z.isRealNumber() << '\n';
      // 1 0

    const B1 b1;
    std::cout << b1.is_B1() << ' ' << b1.is_B2() << '\n'; // 1 0 
  }

  // DerivedClassesHaveTypeProperties
  {
    std::cout << "\n DerivedClassesHaveTypeProperties \n";

    std::cout << "\n AbstractClassesHasTypeProperties \n";

    std::cout << " is_class : " << std::is_class<ImaginaryNumber>::value << '\n'; // 1
    std::cout << " is_class : " << std::is_class<B1>::value << '\n'; // 1

    // compound type (array, function, object ptr, function ptr, member object
    // ptr, member function ptr, reference, class, union, or enum)
    std::cout << " is_compound : " << std::is_compound<ImaginaryNumber>::value << '\n';
      // 1
    std::cout << " is_compound : " << std::is_compound<B1>::value << '\n'; // 1

    // checks if trivially copyable, and has 1 or more default ctors all
    // trivial or deleted, and at least 1 which isn't deleted
    std::cout << " is_trivial : " << std::is_trivial<ImaginaryNumber>::value <<
      '\n'; // 0
    std::cout << " is_trivial : " << std::is_trivial<B1>::value <<
      '\n'; // 0

    // checks if copy, move ctors, copy, move assignment trivial or deleted, at
    // least 1 copy, move ctor, copy, move assignment operator is non-deleted,
    // trivial, non-deleted dtor, implies no virtual.
    std::cout << " is_trivially_copyable : " <<
      std::is_trivially_copyable<ImaginaryNumber>::value << '\n'; // 0
    std::cout << " is_trivially_copyable : " <<
      std::is_trivially_copyable<B1>::value << '\n'; // 0

    // checks if scalar type, standard layout type for communicating with code
    // in other langauges; requires all non-static data members have same
    // access control, no virtual functions or virtual base classes, no
    // non-static data members of reference type, all non-static data members
    // and base classes are themselves standard layout types
    std::cout << " is_standard_layout : " <<
      std::is_standard_layout<ImaginaryNumber>::value << '\n'; // 0
    std::cout << " is_standard_layout : " <<
      std::is_standard_layout<B1>::value << '\n'; // 0

    // checks if type is plain-old data (POD) type
    std::cout << " is_pod : " << std::is_pod<ImaginaryNumber>::value << '\n'; // 0
    std::cout << " is_pod : " << std::is_pod<B1>::value << '\n'; // 0

    // checks if type is class (but not union) and has no data
    std::cout << " is_empty : " << std::is_empty<ImaginaryNumber>::value << '\n'; // 0
    std::cout << " is_empty : " << std::is_empty<B1>::value << '\n'; // 0

    // That is non-union class that declares or inherits at least 1 virtual function
    std::cout << " is_polymorphic : " <<
      std::is_polymorphic<ImaginaryNumber>::value << '\n'; // 1
    std::cout << " is_polymorphic : " <<
      std::is_polymorphic<B1>::value << '\n'; // 1

    std::cout << " is_abstract : " << std::is_abstract<ImaginaryNumber>::value << '\n';
      // 0
    std::cout << " is_abstract : " << std::is_abstract<B1>::value <<
      '\n'; // 0

    std::cout << " is_final : " << std::is_final<ImaginaryNumber>::value << '\n'; // 0
    std::cout << " is_final : " << std::is_final<B1>::value << '\n';
      // 0
  }

  // InitializePointerOfBaseClassToPointerToDerivedClass
  {
    std::cout << "\n InitializePointerOfBaseClassToPointerToDerivedClass \n";
    ImaginaryNumber z;
    Number* a_z {&z};

    // pointer to type Number will dereference to address of z, of type
    // ImaginaryNumber, and will lookup virtual pointer (vptr) of z to vtable
    // of z and to the right function implementation.
    std::cout << " isImaginaryNumber() : " << a_z->isImaginaryNumber() << '\n';
      // 1
    std::cout << " isRealNumber() : " << a_z->isRealNumber() << '\n'; // 0

    B1 b1;
    AAbstract* a_ptr {&b1};

    std::cout << " is_B1() : " << a_ptr->is_B1() << '\n'; // 1
    std::cout << " is_B2() : " << a_ptr->is_B2() << '\n'; // 0

    const AAbstract& a_ref {b1};

    std::cout << " is_B1() : " << a_ref.is_B1() << '\n'; // 1
    std::cout << " is_B2() : " << a_ref.is_B2() << '\n'; // 0
  }

  {
    Employee e {"Brown", 1234};
    Manager m {"Smith", 1234, 2};

    print_list({&e, &m});
  }

  {
    A1 a1 {"a1_1", 42};

    // BA1 ba1 {"ba1_1", 43}; // error: no matching function for call to 
    // BA1::BA1(<brace-enclosed initializer list>)
    BA1 ba1 {"ba1_1", 41, 43};

    // No matching function for call to BA1::BA1();
    // BA1 ba2;

    a1.print();
    ba1.print();

    BA2 ba2;
    ba2.print();

    BA2 ba2_1 {"ba2_1", 44};
    ba2_1.print();

    const A1* a1_ptr {&ba1};
    a1_ptr->print();
    std::cout << " a1_ptr->is_A1() : " << a1_ptr->is_A1() << '\n'; // 0

  }

  // final specifier
  {
    Base base;
    A a;
    B b;
  }
}
