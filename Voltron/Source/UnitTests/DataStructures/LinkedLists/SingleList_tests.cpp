#include "DataStructures/LinkedLists/SingleList.h"

#include <boost/test/unit_test.hpp>

using DataStructures::LinkedLists::DWHarder::SingleList;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(LinkedLists)
BOOST_AUTO_TEST_SUITE(SingleList_tests)

BOOST_AUTO_TEST_CASE(DefaultConstructsOnStack)
{
  SingleList<int> ls {};
}

BOOST_AUTO_TEST_SUITE_END() // SingleList_tests
BOOST_AUTO_TEST_SUITE_END() // LinkedLists
BOOST_AUTO_TEST_SUITE_END() // DataStructures