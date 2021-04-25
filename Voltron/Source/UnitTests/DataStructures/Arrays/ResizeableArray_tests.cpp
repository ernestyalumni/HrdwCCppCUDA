#include "DataStructures/Arrays/ResizeableArray.h"

#include <boost/test/unit_test.hpp>

using DataStructures::Arrays::ResizeableArray;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Arrays)
BOOST_AUTO_TEST_SUITE(ResizeableArray_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsFromInitializerList)
{
  ResizeableArray<int> array {16, 14, 10, 8, 7, 9 , 3, 2, 4, 1};

  BOOST_TEST(array[0] == 16);
  BOOST_TEST(array[1] == 14);  
}

BOOST_AUTO_TEST_SUITE_END() // ResizeableArray_tests
BOOST_AUTO_TEST_SUITE_END() // Arrays
BOOST_AUTO_TEST_SUITE_END() // DataStructures