//------------------------------------------------------------------------------
// \file Category_tests.cpp
//------------------------------------------------------------------------------
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(Categories)
BOOST_AUTO_TEST_SUITE(Category_tests)

// cf. Mac Lane Saunders (1978), pp. 11 Categories

template <typename T>
using TypeAsObject = T;

// Obj(A) for category A
template <typename T>
using TypeAsObjects = T;

// TODO: provide class implementation for classes and structs.
//template <typename TypeAsObjects>
//class Object
//{
//  Object 

//};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(NonegativeIntegersAsCategoryObject)
{
  using UnsignedIntObject = TypeAsObject<unsigned int>;

  const UnsignedIntObject a {42};

  BOOST_TEST(a == 42);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(NonegativeIntegersAsCategoryObjects)
{
  using UnsignedIntObjects = TypeAsObjects<unsigned int>;

  const UnsignedIntObjects a {42};

  BOOST_TEST(a == 42);
}


BOOST_AUTO_TEST_SUITE_END() // Category_tests
BOOST_AUTO_TEST_SUITE_END() // Categories