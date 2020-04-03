//------------------------------------------------------------------------------
/// \file PointersArraysReferences_tests.cpp
/// \ref Bjarne Stroustrup. The C++ Programming Language, 4th Edition.
/// Addison-Wesley Professional. May 19, 2013. ISBN-13: 978-0321563842
//------------------------------------------------------------------------------
#include <boost/test/unit_test.hpp>
#include <cstring>
#include <iostream>
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

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(NullptrAssignedToAnyPointerType)
{
  int* pi = nullptr;
  int* pi2 {nullptr};
  double* pd = nullptr;
  //int i = nullptr; // error: i is not a pointer.
  BOOST_TEST(true);
}

// cf. 7.3 Arrays, pp. 174, Stroustrup (2013)
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DeclareAndAssignToArrays)
{
  float v[3]; // array of 3 floats
  char* a[32]; // array of 32 pointers to char

  int aa[10];
  aa[6] = 9; // assign to aa's 7th element

  BOOST_TEST(true);
}

// cf. 7.3.1 Array Initializers, pp. 175, Stroustrup (2013)
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ArraysCanBeInitialized)
{
  int v1[] = {1, 2, 3, 4};
  char v2[] = {'a', 'b', 'c', 0};

  // char v3[2]= {'a', 'b', 0}; // error: too many initializers
  char v4[3] = {'a', 'b', 0};

  // If initializer supplies too few elements for an array, 0 used for the rest
  int v5[8] = {1, 2, 3, 4};
  BOOST_TEST(v5[0] == 1);
  BOOST_TEST(v5[1] == 2);
  BOOST_TEST(v5[2] == 3);
  BOOST_TEST(v5[3] == 4);
  BOOST_TEST(v5[4] == 0);
  BOOST_TEST(v5[5] == 0);
  BOOST_TEST(v5[6] == 0);
  BOOST_TEST(v5[7] == 0);
}

const char* error_message_returning_str_literal(int i)
{
  i + 1;
  return "range error";
}
// cf. 7.3.2 String Literals, pp. 176, Stroustrup (2013)
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StringLiteralsInitializeArrays)
{
  "this is a string";

  // A string literal contains 1 more character than it appears to have; it's
  // terminated by null character, '\0', with value 0

  BOOST_TEST(sizeof("Bohr") == 5);

  // char* p ="Plato"; // error, C++ forbids converting string constant to char*

  // If we want a string we are guranteed to be able to modify, we must place
  // characters in a non-const array

  char p[] = "Zeno"; // p is an array of 5 char
  BOOST_TEST(sizeof(p) == 5);
  p[0] = 'R'; // OK
  BOOST_TEST(std::string {p} == "Reno");

  BOOST_TEST(std::string{error_message_returning_str_literal(5)} ==
    "range error");

  // Long strings can be broken by whitespace to make the program text neater.
  char alpha[] = "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

  BOOST_TEST(std::string{alpha} ==
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ");
}

// cf. 7.3.2.1 Raw Character Strings, pp. 177, Stroustrup (2013)
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(RawCharacterStrings)
{
  std::string s {R"(\w\\w)"};

  BOOST_TEST(s == R"(\w\\w)");

  // "( and )" is the only default delimiter pair.

  s = R"***("quoted string containing the usual terminator ("))")***";
  BOOST_TEST(s == "\"quoted string containing the usual terminator (\"))\"");

  std::string counts {R"(1
22
333)"};

  std::string x {"1\n22\n333"};

  BOOST_TEST(counts == x);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ImplicitConversionFromArrayToPointer)
{
  int v[] = {1, 2, 3, 4};
  int* p1 = v; // pointer to initial element (implicit conversion)
  int* p2 = &v[0]; // pointer to initial element
  int* p3 = v + 4; // pointer to one beyond-last element

  BOOST_TEST(*p1 == 1);
  BOOST_TEST(*p2 == 1);

  {
    char v[] = "Annemarie";
    char* p = v; // implicit conversion of char[] to char*
    BOOST_TEST(strlen(p) == 9);
    BOOST_TEST(strlen(v) == 9); // implicit conversion of char[] to char*
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(NavigateArraysByPointerArithmetic)
{
  char v[] {"123456789"};
  BOOST_TEST_REQUIRE(sizeof(v) == 10);
  for (int i {0}; v[i] != 0; ++i)
  {
    BOOST_TEST(v[i] == static_cast<char>(i + '1'));
  }

  char p[] {"abcdefghijklmnopqrstuvwxyz"};
  BOOST_TEST_REQUIRE(sizeof(p) == 27);
  int i {0};
  for (char* ptr = p; *ptr != 0; ++ptr)
  {
    BOOST_TEST(*ptr == static_cast<char>('a' + i));
    ++i;
  }

  // The prefix * operator dereferences a pointer so that *p is the character
  // pointed to by p, and ++ increments the pointer so that it refers to the
  // next element of the array.

  const char a[] {"ABCDEFGH"} ; 

  constexpr int j {2};

  BOOST_TEST(a[j] == *(&a[0] + j)); 
  BOOST_TEST(*(&a[0] + j) == *(a+j));
  BOOST_TEST(*(a+j) == *(j + a));
  BOOST_TEST(*(j + a) == j[a]);
  BOOST_TEST(3["Texas"] == "Texas"[3]);
  BOOST_TEST("Texas"[3] == 'a');
}

template <typename T>
int byte_diff(T* p, T* q)
{
  return reinterpret_cast<char*>(q) - reinterpret_cast<char*>(p);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ApplySubtractionForPointerArithmetic)
{
  std::cout << "\n ApplySubtractionForPointerArithmetic \n";

  int vi[10];
  short vs[10];

  // Pointer values printed using default hexadecimal notation.
  // e.g. 
  // 0x7ffe36969950 0x7ffe36969954
  // 0x7ffe36969930 0x7ffe36969932
  std::cout << vi << ' ' << &vi[1] << '\n';
  std::cout << vs << ' ' << &vs[1] << '\n';

  // Result is number of array elements in the sequence [p:q) (an integer).
  BOOST_TEST((&vi[1] - vi) == 1);
  BOOST_TEST((&vs[1] - vs) == 1);
  BOOST_TEST((&vi[1] - &vi[0]) == 1);
  BOOST_TEST((&vs[1] - &vs[0]) == 1);

  BOOST_TEST(byte_diff(&vi[0], &vi[1]) == 4);
  BOOST_TEST(byte_diff(&vs[0], &vs[1]) == 2);

  int v1[10];
  int v2[10];

  int i1 = &v1[5] - &v1[3];
  BOOST_TEST(i1 == 2);
  // int i2 = &v1[5] - &v2[3]; // result undefined

  int* p1 = v2 + 2; // p1 - &v2[2]
  // int*p2 = v2 - 2 // *p2 undefined
}

void fp(char v[], int size)
{
  for (int i {0}; i != size; ++i)
  {
    BOOST_TEST(v[i] == i + 'a');
  }

  // for (int x : v)
  //  use(x); // error: range-for does not work for pointers

  constexpr int N {7};
  char v2[N];
  for (int i {0}; i != N; ++i)
  {
    v2[i];
  }
  for (int x : v2)
  {
    x; // range-for works for arrays of known size.
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ArrayTraverseRequiresExplicitlyStatedSize)
{
  char v[] {"abcdefgh"};
  int size {8};
  fp(v, size);
  BOOST_TEST(true);
}

BOOST_AUTO_TEST_SUITE_END() // PointersArraysReferences_tests
BOOST_AUTO_TEST_SUITE_END() // PointersArraysReferences
BOOST_AUTO_TEST_SUITE_END() // Cpp