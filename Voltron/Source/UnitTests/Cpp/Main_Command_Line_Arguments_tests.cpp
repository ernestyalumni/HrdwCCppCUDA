//------------------------------------------------------------------------------
/// \file Main_Command_Line_tests.cpp
/// \ref Bjarne Stroustrup. The C++ Programming Language, 4th Edition.
/// Addison-Wesley Professional. May 19, 2013. ISBN-13: 978-0321563842
//------------------------------------------------------------------------------
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Main_Command_Line_Arguments_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TemplateArgumentDeducedAsPartOfConstReferenceType)
{
  std::complex<float> c1, c2;
  c1 = {3.0, -5.0};
  c2 = {42.0, -69.0};

  //::max(c1, c2); // ERROR at compile time.

  BOOST_TEST(true);
}

BOOST_AUTO_TEST_SUITE_END() // Main_Command_Line_Arguments_tests
BOOST_AUTO_TEST_SUITE_END() // Cpp