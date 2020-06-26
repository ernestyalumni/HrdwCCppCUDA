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
BOOST_AUTO_TEST_CASE(PointerDeclarations)
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

// Remember, reinterpret_cast is resolved at compile-time; it's nothing more
// than "look to a pointer that's pointing to type A with eyes of who is
// looking for type B".
// https://stackoverflow.com/questions/27309604/do-constant-and-reinterpret-cast-happen-at-compile-time/27309763
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

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DeclareMultidimensionalArrays)
{
  int ma[3][5]; // 3 arrays with 5 ints each

  BOOST_TEST(true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(InitializeMultidimensionalArrays)
{
  int ma[3][5]; // 3 arrays with 5 ints each

  for (int i {0}; i != 3; ++i)
  {
    for (int j {0}; j != 5; ++j)
    {
      ma[i][j] = 10 * i + j;
    }
  }

  for (int k {0}; k < 15; ++k)
  {
    BOOST_TEST(
      ma[ k / 5][ k - 5 * (k / 5)] == 10 * (k / 5) + (k - 5 * (k / 5)));
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MultidimensionalArraysAreRowMajor)
{
  int ma[3][5]; // 3 arrays with 5 ints each

  // Initialize array.
  for (int i {0}; i != 3; ++i)
  {
    for (int j {0}; j != 5; ++j)
    {
      ma[i][j] = 10 * i + j;
    }
  }

  for (int i {0}; i < 2; ++i)
  {
    // size of int (4 bytes) * 5 elements in a "row"
    BOOST_TEST(byte_diff(&ma[i], &ma[i + 1]) == 20);
  }

  for (int i {0}; i < 3; ++i)
  {
    for (int j {0}; j < 4; ++j)
    {
      BOOST_TEST(byte_diff(&ma[i][j], &ma[i][j + 1]) == 4);
    }
  }

  // Need 2nd. dimension to locate actual first element.
  int* ptr {&ma[0][0]};

  for (int k {0}; k < 15; ++k)
  {
    BOOST_TEST(*(ptr + k) == 10 * (k / 5) + (k - 5 * (k / 5)));
  }
}

// cf. 7.4.3 Passing Arrays, pp. 184, Stroustrup (2013)

void comp(double arg[10]) // arg is a double*
{
  for (int i {0}; i != 10; ++i)
  {
    arg[i] += 99;
  }
}

// This function is equivalent to comp.
void comp2(double* arg) // arg is a double*
{
  for (int i {0}; i != 10; ++i)
  {
    arg[i] += 99;
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PassArraysAsPointerToFirstElement)
{
  double a1[10];
  double a2[5];
  double a3[100];

  comp(a1);
  // The following line comp(a2) compiles
  // comp(a2); // disaster!
  comp(a3); // uses only the first 10 elements

  for (int i {0}; i < 10; ++i)
  {
    a1[i] == 99;
    a3[i] == 99;
  }

  comp2(a1);
  comp2(a3);

  for (int i {0}; i < 10; ++i)
  {
    a1[i] == 189;
    a3[i] == 189;
  }
}

// cf. 7.4.3 Passing Arrays, pp. 184-185, Stroustrup (2013)
// If dimensions are known at compile time, passing arrays as pointer.

int expected_v[3][5] {
  {0, 1, 2, 3, 4},
  {10, 11, 12, 13, 14},
  {20, 21, 22, 23, 24}
};

void print_m35(int m[3][5])
{
  for (int i {0}; i != 3; ++i)
  {
    for (int j {0}; j != 5; ++j)
    {
      BOOST_TEST(m[i][j] == expected_v[i][j]);
    }
  }
}

void print_mi5(int m[][5], int dim1)
{
  for (int i {0}; i != dim1; ++i)
  {
    for (int j {0}; j != 5; ++j)
    {
      BOOST_TEST(m[i][j] == expected_v[i][j]);
    }
  }
}

// argument declaration m[][] is illegal because 2nd. dimension of a
// multidimensional array must be known in order to find location of an element.
//void print_mij(int m[][], int dim1, int dim2)
// To call this function, we pass a matrix as an ordinary pointer.
void print_mij(int* m, int dim1, int dim2)
{
  for (int i {0}; i != dim1; ++i)
  {
    for (int j {0}; j != dim2; ++j)
    {
      BOOST_TEST(m[i * dim2 + j] == expected_v[i][j]);
    }
  }
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PassArraysAsPointerIfDimensionsKnown)
{
  int v[3][5] {
    {0, 1, 2, 3, 4},
    {10, 11, 12, 13, 14},
    {20, 21, 22, 23, 24}
  };

  print_m35(v);
  print_mi5(v, 3);

  print_mij(&v[0][0], 3, 5);
}

// cf. Sec. 7.5, Pointers and const, Stroustrup (2013), pp. 186

void f1(char* p)
{
  char s[] = "Gorm";
  const char* pc = s; // pointer to constant

  const char* pc2 {s}; 
  // pc[3] = 'g'; // error: pc points to constant

  pc = p; // OK
  
  char* const cp = s; // constant pointer
  cp[3] = 'a'; // OK
  // cp = p; // error: cp is constant

  const char* const cpc = s; // const pointer to const
  //cpc[3] = 'a'; // error: cpc points to constant
  //cpc = p; // error: cpc is constant  

}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DeclareWithConst)
{
  const int model = 90; // mode is a const
  const int v[] = {1, 2, 3, 4}; // v[i] is a const
  //const int x; // error; no initializer

  char* p;

  f1(p);

  // error: uninitalized const
  //char* const cp; // const pointer to char
  char* const cp {p};

  char const* pc; // pointer to const char
  const char* pc2; // pointer to const char

  BOOST_TEST(true);
}

// This 1st version is used for strings where elements mustn't be modified and
// returns a pointer to const that does not allow modification.
const char* strchr(const char* p, char c); // find first occurrence of c in p

// 2nd version used for mutable strings
char* strchr(char* p, char c);

BOOST_AUTO_TEST_SUITE(References)

void f(std::vector<double>& v)
{
  double d1 = v[1]; // copy the value of the double referred to by
  // v.operator[](1) into d1
  v[2] = 7;

  v.push_back(d1); // give push_back() a reference to d1 to work with
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ReferenceUsedToSpecifyArguments)
{
  std::vector<double> v {0.0, 1.2, 1.3};
  BOOST_TEST_REQUIRE(v.size() == 3);
  f(v);
  BOOST_TEST(v[0] == 0.0);
  BOOST_TEST(v[1] == 1.2);
  BOOST_TEST(v[2] == 7);
  BOOST_TEST(v[3] == 1.2);  
  BOOST_TEST(v.size() == 4);
}

BOOST_AUTO_TEST_SUITE_END() // References

BOOST_AUTO_TEST_SUITE_END() // PointersArraysReferences_tests
BOOST_AUTO_TEST_SUITE_END() // PointersArraysReferences
BOOST_AUTO_TEST_SUITE_END() // Cpp