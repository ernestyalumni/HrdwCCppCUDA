//------------------------------------------------------------------------------
// \file AccessAndInheritance_test.cpp
//------------------------------------------------------------------------------
#include "Hierarchy/AccessAndInheritance.h"

#include <boost/test/unit_test.hpp>
#include <iostream>

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


BOOST_AUTO_TEST_SUITE_END() // Access_tests
BOOST_AUTO_TEST_SUITE_END() // Utilities
