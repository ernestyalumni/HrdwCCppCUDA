//------------------------------------------------------------------------------
/// \file TypeUtilities_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://gitlab.com/manning-fpcpp-book/code-examples/-/blob/master/chapter-11/contained-type/type_utils.h
//------------------------------------------------------------------------------
#include "Cpp/Templates/TypeUtilities.h"

#include <boost/test/unit_test.hpp>
#include <list>
#include <string>
#include <type_traits> // std::is_same, std::remove_reference
#include <utility> // std::declval
#include <vector>

using Cpp::Templates::TypeUtilities::ContainedType;
using Cpp::Templates::TypeUtilities::Error;

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Templates)
BOOST_AUTO_TEST_SUITE(TypeUtilities_tests)

struct Default
{
  int foo() const
  {
    return 1;
  }
};

struct NonDefault
{
  NonDefault() = delete;
  int foo() const
  {
    return 1;
  }
};

// cf. https://en.cppreference.com/w/cpp/utility/declval
// std::declval
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StdDeclvalConvertsAnyTypeWithoutDefaultConstructor)
{
  decltype(Default().foo()) n1 {1}; // type of n1 is int
  // decltype(NonDefault().foo()) n2 = n1; // error: no default constructor
  decltype(std::declval<NonDefault>().foo()) n2 {n1}; // type of n2 is int
  BOOST_TEST(n1 == 1);
  BOOST_TEST(n2 == 1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StdRemoveReferenceProvidesTypeReferredTo)
{
  BOOST_TEST((std::is_same<int, int>()));
  BOOST_TEST(!(std::is_same<int, int&>()));
  BOOST_TEST(!(std::is_same<int, int&&>()));

  BOOST_TEST((std::is_same<int, std::remove_reference_t<int>>()));
  BOOST_TEST((std::is_same<int, std::remove_reference_t<int&>>()));
  BOOST_TEST((std::is_same<int, std::remove_reference_t<int&&>>()));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StdRemoveCvRemovesConstVolatileQualifiersFromTypes)
{
  typedef std::remove_cv_t<const int> type1;
  typedef std::remove_cv_t<volatile int> type2;
  typedef std::remove_cv_t<const volatile int> type3;

  BOOST_TEST((std::is_same<int, type1>()));
  BOOST_TEST((std::is_same<int, type2>()));
  BOOST_TEST((std::is_same<int, type3>()));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StdRemoveCvRemovesConstVolatileQualifiersFromPointerTypes)
{
  typedef std::remove_cv_t<const volatile int*> type4;
  typedef std::remove_cv_t<int* const volatile> type5;

  BOOST_TEST((std::is_same<const volatile int*, type4>()));
  BOOST_TEST(!(std::is_same<int*, type4>()));
  BOOST_TEST((std::is_same<int*, type5>()));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ContainedTypeGetsTypeOfContainerElements)
{
  BOOST_TEST((
    std::is_same<
      ContainedType<std::vector<std::string>>,
      std::string
      >()
    ));

  BOOST_TEST((
    std::is_same<
      NonDefault,
      ContainedType<std::vector<NonDefault>>
      >()
    ));
}

// Writes the exact result of the ContainedType meta-function
// Error<ContainedType<std::vector<std::string>>>();
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ErrorProducesCompilationError)
{
  // Produces compilation error, is not a template.
  //Error<ContainedType<std::vector<std::string>>>();
  BOOST_TEST(true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StaticAssertVerifiesContainedType)
{
  static_assert(
    std::is_same<int, ContainedType<std::vector<int>>>(),
    "std::vector<int> should contain integers");

  static_assert(
    std::is_same<
      std::string,
      ContainedType<
        std::list<std::string>>>(),
    "std::list<std::string> should contain strings");

  static_assert(
    std::is_same<
      NonDefault*,
      ContainedType<
        std::vector<NonDefault*>>>(),
    "std::vector<NonDefault*> should contain NonDefault*");

  BOOST_TEST(true);
}

BOOST_AUTO_TEST_SUITE_END() // TypeUtilities_tests 

BOOST_AUTO_TEST_SUITE_END() // Templates
BOOST_AUTO_TEST_SUITE_END() // Cpp