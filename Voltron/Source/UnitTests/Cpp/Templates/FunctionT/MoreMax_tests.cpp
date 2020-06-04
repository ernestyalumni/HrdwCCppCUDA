//------------------------------------------------------------------------------
/// \file Max_tests.cpp
/// \ref Vandevoorde, Josuttis, Gregor. C++ Templates: The Complete Guide. 2nd
/// Ed. Addison-Wesley Professional. 2017.
//------------------------------------------------------------------------------
#include "Cpp/Templates/FunctionT/MoreMax.h"

#include <boost/test/unit_test.hpp>
#include <complex>
#include <string>

using namespace Cpp::Templates::FunctionTemplates;

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Templates)
BOOST_AUTO_TEST_SUITE(FunctionTemplates)
BOOST_AUTO_TEST_SUITE(Max_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ShowHowToUseFunctionTemplateDefinition)
{
  // Templates aren't compiled into single entities that can handle any type.
  // Instead, different entities are generated from the template for every type
  // for which template is used.
  // Process of replacing template parameters by concrete types is called
  // instantiations. Results in an instance of a template.
  // Note mere use of a function template can trigger such an instantiation
  // process.
  constexpr int i {42};
  BOOST_TEST(max1::max(7, i) == i);

  constexpr double f1 {3.4};
  constexpr double f2 {-6.7};
  BOOST_TEST(max1::max(f1, f2) == f1);

  const std::string s1 {"mathematics"};
  const std::string s2 {"math"};
  BOOST_TEST(max1::max(s1, s2) == s1);
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TemplateArgumentDeducedAsPartOfConstReference)
{
  {
    int c {42};
    int i {7};

    BOOST_TEST(max_with_const_refs::max(i, c) == c);
  }

  {
    BOOST_TEST(max_with_const_refs::max(2, 3) == 3);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CompilerDeducesReturnTypeWithAuto)
{
  {
    const int c {42};
    const double i {7.0};

    BOOST_TEST(max_auto::max(i, c) == c);
  }

  {
    BOOST_TEST(max_auto::max(2.0, 3) == 3);
  }
}

// Trailing return type is ->
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(Cpp11UsesTrailingReturnType)
{
  {
    const int c {42};
    const double i {7.0};

    BOOST_TEST(max_decltype::max(i, c) == c);
  }

  {
    BOOST_TEST(max_decltype::max(2.0, 3) == 3);
  }
}


/*


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TypeDeductionForDefaultTemplateArgument)
{
  f1(); // OK
  BOOST_TEST(true);
}

// pp. 15, Sec. 1.5 Overloaindg Function Templates
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(OverloadingFunctionTemplates)
{
  BOOST_TEST(max(7, 42) == 42); // calls nontemplate for 2 ints.
  BOOST_TEST(max(7.0, 42.0) == 42.0); // call max <double> (by argument
    //deduction)
  BOOST_TEST(max('a', 'b') == 'b'); // calls max<char> (by argument deduction)
  BOOST_TEST(max<>(7, 42) == 42); // call max<int> (by argument deduction)
  BOOST_TEST(max<double>(7, 42) == 42.0); // calls max<double> (no argument
  // deduction)
  BOOST_TEST(max('a', 42.7) == 97); // call the nontemplate for two ints 
}
*/

BOOST_AUTO_TEST_SUITE_END() // Max_tests

BOOST_AUTO_TEST_SUITE_END() // FunctionTemplates
BOOST_AUTO_TEST_SUITE_END() // Templates
BOOST_AUTO_TEST_SUITE_END() // Cpp