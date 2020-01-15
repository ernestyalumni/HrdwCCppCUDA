//------------------------------------------------------------------------------
// \file Keywords_tests.cpp
//------------------------------------------------------------------------------

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Keywords_tests)

// cf. https://www.tutorialspoint.com/where-are-static-variables-stored-in-c-cplusplus

int func()
{
  static int i {4};
  i++;
  return i;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// cf. https://www.tutorialspoint.com/where-are-static-variables-stored-in-c-cplusplus
BOOST_AUTO_TEST_CASE(DemonstrateStaticVariables)
{
  // In func(), is a static variable, stored in initialized data segment.
  // In main() function, func() returns static variable; it remains in memory
  // because it's static while program is running and provides consistent
  // values.
  {
    BOOST_TEST(func() == 5);
    BOOST_TEST(func() == 6);
    BOOST_TEST(func() == 7);
    BOOST_TEST(func() == 8);
    BOOST_TEST(func() == 9);
    BOOST_TEST(func() == 10);
  }
  BOOST_TEST(func() == 11);
  BOOST_TEST(func() == 12);
  BOOST_TEST(func() == 13);
}

BOOST_AUTO_TEST_SUITE_END() // Keywords_tests
BOOST_AUTO_TEST_SUITE_END() // Cpp