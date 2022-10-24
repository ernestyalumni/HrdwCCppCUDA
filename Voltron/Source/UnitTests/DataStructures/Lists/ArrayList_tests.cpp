#include "DataStructures/Lists/ArrayList.h"

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Lists)
BOOST_AUTO_TEST_SUITE(Shaffer)
BOOST_AUTO_TEST_SUITE(ArrayList_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
  ArrayList<char> array_list {};

  BOOST_TEST(true);
}

BOOST_AUTO_TEST_SUITE_END() // ArrayList_tests
BOOST_AUTO_TEST_SUITE_END() // Shaffer
BOOST_AUTO_TEST_SUITE_END() // Lists
BOOST_AUTO_TEST_SUITE_END() // DataStructures