//------------------------------------------------------------------------------
/// \file Max_tests.cpp
/// \ref Vandevoorde, Josuttis, Gregor. C++ Templates: The Complete Guide. 2nd
/// Ed. Addison-Wesley Professional. 2017.
//------------------------------------------------------------------------------
#include "Cpp/Templates/FunctionT/Max.h"

#include <boost/test/unit_test.hpp>
#include <complex>
#include <string>

using Cpp::Templates::FunctionTemplates::max;
using Cpp::Templates::FunctionTemplates::max_with_const;

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Templates)
BOOST_AUTO_TEST_SUITE(FunctionTemplates)
BOOST_AUTO_TEST_SUITE(Max_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateMax)
{
	{
		// Works for ints
		const int a {4};
		const int b {5};
		BOOST_TEST(::max(a, b) == b);
	}
}

// cf. VJG (2017), pp. 4, 1.1.2 Using the Template
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ShowHowToUseMax)
{
  constexpr int i {42};

  // Note that each call of max() template qualified with ::. This is to ensure
  // that our max() template is found in the global namespace. There is also a
  // std::max() template in standard library.
  BOOST_TEST(::max(7, i) == i);

  constexpr double f1 {3.4};
  constexpr double f2 {-6.7};

  BOOST_TEST(::max(f1, f2) == f1);

  const std::string s1 {"mathematics"};
  const std::string s2 {"math"};

  // The process of replacing template parameters by concrete types is called
  // instantiation. It results in an instance of a template.
  BOOST_TEST(::max(s1, s2) == s1);

  // Templates aren't compiled into single entities that can handle any type.
  // Instead, different entities are generated from template for every type for
  // which template is used. Thus, max() compiled for each of these 3 types.
}

// void is a valid template argument.
template <typename T>
T foo(T*)
{  
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(VoidIsAValidTemplateArgument)
{
  void* vp = nullptr;

  foo(vp); // OK: deduces void foo(void*)

  BOOST_TEST(true);
}

// cf. pp. 6, 1.1.3 Two-Phase Translation, VJG (2017)
// An attempt to instantiate a template for type that doesn't support all the
// operations used within it will result in compile-time error.


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(InstantiateWithTypeThatDoesNotSupportAllOperations)
{
  std::complex<float> c1, c2;
  c1 = {3.0, -5.0};
  c2 = {42.0, -69.0};

  //::max(c1, c2); // ERROR at compile time.

  BOOST_TEST(true);
}

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

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TypeConversionsDuringTypeDeductionByValueDecay)
{
  constexpr int c {42};
  int i {64};
  BOOST_TEST_REQUIRE(max(i, c) == i); // OK: T is deduced as int
  BOOST_TEST_REQUIRE(max(i, c) == i); // OK: T is deduced as int

  int& ir {i};
  BOOST_TEST_REQUIRE(max(i, ir) == i); // OK: T is deduced as int
  int arr[4];
  BOOST_TEST(max(&i, arr) == &i);

  // ERROR: T can be deduced as int or double.
  //max(4, 7.2);

  std::string s;

  // ERROR: T can be deduced as char const[6] or std::string
  //max("hello", s);

  // 3 ways to handle such errors:

  // 1. Cast the arguments so that they both match:
  BOOST_TEST(max(static_cast<double>(4), 7.2) == 7.2); // OK

  // 2. Specify (or qualify) explicitly the type of T to prevent from attempting
  // type deduction:
  BOOST_TEST(max<double>(4, 7.2) == 7.2);

  BOOST_TEST(true);
}

BOOST_AUTO_TEST_SUITE_END() // Max_tests
BOOST_AUTO_TEST_SUITE_END() // FunctionTemplates
BOOST_AUTO_TEST_SUITE_END() // Templates
BOOST_AUTO_TEST_SUITE_END() // Cpp