//------------------------------------------------------------------------------
// \file Operations_test.cpp
//------------------------------------------------------------------------------

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Operations_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// cf. Stroustrup. C++ Programming Language, 4th. Ed. pp. 302 Sec. 11.5.2
BOOST_AUTO_TEST_CASE(PointersToFunctionsStaticCastReinterpretCasts)
{
  char x = 'a';
  char xb {'a'};

  // error: cannot convert char* to int* in initialization
  //int* p1 = &x; // error: no implicit char* to int* conversion

  //  int* p1b {&x};
  // invalid static_cast from type char* to type int*
  //int* p2 = static_cast<int*>(&x); // error: no implicit char* to int* conversion
  int* p3 = reinterpret_cast<int*>(&x); // OK: on your head be it

  struct B
  {
    int b;
  };

  struct D : B
  {
    int d;
  };

  B* pb = new D; // OK: implicit conversion from D* to B*
  //D* pd = pb;

  // compile error: error invalid conversion
  //D* pd = pb; // error: no implicit conversion from B* to D*

  D* pd = static_cast<D*>(pb); // OK

  delete pb;

  BOOST_TEST(true);
}

BOOST_AUTO_TEST_SUITE_END() // Operations_tests
BOOST_AUTO_TEST_SUITE_END() // Cpp