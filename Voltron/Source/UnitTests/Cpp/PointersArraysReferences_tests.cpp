//------------------------------------------------------------------------------
/// \file PointersArraysReferences_tests.cpp
/// \ref Bjarne Stroustrup. The C++ Programming Language, 4th Edition.
/// Addison-Wesley Professional. May 19, 2013. ISBN-13: 978-0321563842
//------------------------------------------------------------------------------
#include <boost/test/unit_test.hpp>
#include <string>

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(PointersArraysReferences)
BOOST_AUTO_TEST_SUITE(PointersArraysReferences_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(Declarations)
{
  // cf. Stroustrup (2013). 6.3.1, pp. 153. Definition of an array of C-style
  // strings.

  // Postfix declarator, e.g. [], bind tighter than prefix ones, e.g. *
  // array of pointers to char
  const char* kings[] = {"Antigonus", "Seleucus", "Ptolemy"};

  BOOST_TEST(kings[0] == "Antigonus");
  BOOST_TEST(kings[1] == "Seleucus");
  BOOST_TEST(kings[2] == "Ptolemy");

  // Pointer to an array of `char`
  char(*one_king)[5];

  char a_king[] = {'L', 'o', 'u', 'i', 's'};

  one_king = &a_king;

  BOOST_TEST((*one_king)[0] == 'L');
  BOOST_TEST((*one_king)[1] == 'o');
  BOOST_TEST((*one_king)[2] == 'u');
  BOOST_TEST((*one_king)[3] == 'i');
  BOOST_TEST((*one_king)[4] == 's');

  int* pi; // pointer to int
  char** ppc; // pointer to pointer to char
  int* ap[15]; // array of 15 pointers to ints
  int (*fp)(char*); // pointer to function taking a char* argument; returns an
  // int
  int* f(char*); // function taking a char* argument; returns a pointer to int

  BOOST_TEST(true);
}

void void_pointer_points(int* pi)
{
  void* pv = pi; // OK: implicit conversion of int* to void*
  // *pv; // error: can't deference void*
  // ++pv; // error: can't increment void* (the size of the object pointed to is
  // unknown)

  int* pi2 = static_cast<int*>(pv); // explicit conversion back to int*

  // double* pd1 = pv; // error
  // double* pd2 = pi; // error
  // double* pd3 = static_cast<double*>(pv); // unsafe (Sec. 11.5.2)
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(VoidPointerExplicitlyConvertsToAnotherPointer)
{
  {
    const int pi_value {42};
    const int* pi = &pi_value;
    BOOST_TEST(true);
  }
  int pi_value {42};
  int* pi = &pi_value;

  void_pointer_points(pi);
}

BOOST_AUTO_TEST_SUITE_END() // PointersArraysReferences_tests
BOOST_AUTO_TEST_SUITE_END() // PointersArraysReferences
BOOST_AUTO_TEST_SUITE_END() // Cpp