// \file FunctionObject_test.cpp

#include <boost/test/unit_test.hpp>
#include <iostream>

BOOST_AUTO_TEST_SUITE(Std)
BOOST_AUTO_TEST_SUITE(FunctionObject_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StdFunctionIsFun)
{
  std::cout << "\n StdFunctionIsFun\n";

  BOOST_TEST(true);
}

BOOST_AUTO_TEST_SUITE_END() // FunctionObject_tests
BOOST_AUTO_TEST_SUITE_END() // Std
