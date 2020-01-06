//------------------------------------------------------------------------------
// \file AccessAndInheritance_test.cpp
//------------------------------------------------------------------------------
#include "Hierarchy/AccessAndInheritance.h"

#include <boost/test/unit_test.hpp>
#include <iostream>

using Hierarchy::Access::ExPrivateVirtualMethod;
using Hierarchy::Access::ExPublicUsing;
using Hierarchy::Access::ExPublicVirtualMethod;
using Hierarchy::Access::ExPublicVsPrivate;

BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(Access_tests)


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PublicVsPrivateAccessCheckedAtCompileTime)
{
  ExPublicVsPrivate e;
  e.add(1); // OK: public ExPublicVsPrivate::add can be accessed from main.

  BOOST_TEST(true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// This error is obtained at COMPILE-TIME; error:
// 'int Hierarchy::Access::ExPublicVsPrivate::n_' is private within this context
#ifdef FORCE_COMPILE_ERRORS
BOOST_AUTO_TEST_CASE(PrivateMemberCannotBeAccessedFromMain)
{
  ExPublicVsPrivate e;
  e.n_ = 7;

  BOOST_TEST(true);
}
#endif // FORCE_COMPILE_ERRORS

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(UsingNameCheckedNotOriginOfNameReferredTo)
{
  ExPublicUsing::BB x; // OK, ExPublicUsing::BB is public

  BOOST_TEST(true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
#ifdef FORCE_COMPILE_ERRORS
BOOST_AUTO_TEST_CASE(PrivateNestedClassCannotBeAccessedFromMain)
{
  ExPublicUsing::B x; // error, ExPublicUsing is private

  BOOST_TEST(true);
}
#endif // FORCE_COMPILE_ERRORS

// https://en.cppreference.com/w/cpp/language/access
// Access rules for names of virtual functions checked at call point using the
// type of expression used to denote object for which member function is called.
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(InvokePrivateVirtualFunctionFromPublicType)
{
  ExPrivateVirtualMethod private_virtual_method;

  ExPublicVirtualMethod& public_virtual_method = private_virtual_method;

  // OK: ExPublicVirtualMethod::f() is public,
  // ExPrivateVirtualMethod::f() is invoked even though it's private.
  BOOST_TEST(public_virtual_method.f() == 43); // Amazing!

  ExPublicVirtualMethod public_virtual_method_1;

  BOOST_TEST(public_virtual_method_1.f() == 42);

  // cf. https://www.geeksforgeeks.org/what-happens-when-more-restrictive-access-is-given-in-a-derived-class-method-in-c/

  ExPublicVirtualMethod* ptr = new ExPrivateVirtualMethod;
  BOOST_TEST(ptr->f() == 43);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
#ifdef FORCE_COMPILE_ERRORS
BOOST_AUTO_TEST_CASE(InvokePrivateVirtualFunctionErrorsAtCompileTime)
{
  ExPrivateVirtualMethod private_virtual_method;

  BOOST_TEST(private_virtual_method.f() == 42);

  BOOST_TEST(true);
}
#endif // FORCE_COMPILE_ERRORS

// A name that's accessible through multiple paths in inheritance graph has the
// access of path with most access:
class PublicAccessW
{
  public:

    void f();
};

void PublicAccessW::f()
{
  return;
}

class PrivateVirtualInheritanceA : private virtual PublicAccessW
{};

class PublicVirtualInheritanceB : public virtual PublicAccessW
{};

class PublicInheritanceC :
  public
    PrivateVirtualInheritanceA,
    PublicVirtualInheritanceB
{
  public:
    void f()
    {
      PublicAccessW::f();
    }
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PublicAccessInheritance)
{
  PublicAccessW w;
  w.f();

  PublicVirtualInheritanceB b;
  b.f();

  PublicInheritanceC c;
  c.f();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
#ifdef FORCE_COMPILE_ERRORS
BOOST_AUTO_TEST_CASE(PrivateAccessInheritance)
{
  PrivateVirtualInheritanceA a;
  a.f();

  BOOST_TEST(true);
}
#endif // FORCE_COMPILE_ERRORS



BOOST_AUTO_TEST_SUITE_END() // Access_tests
BOOST_AUTO_TEST_SUITE_END() // Utilities
