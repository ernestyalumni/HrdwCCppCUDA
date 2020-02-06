//------------------------------------------------------------------------------
/// \file Entrevue_tests.cpp
///
/// \ref cf. https://www.bogotobogo.com/cplusplus/functors.php
//------------------------------------------------------------------------------

#include <boost/test/unit_test.hpp>
#include <algorithm> // std::for_each
#include <iostream>
#include <vector>


BOOST_AUTO_TEST_SUITE(Entrevue)
BOOST_AUTO_TEST_SUITE(Entrevue_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(Demonstrate)
{
  std::cout << "\n\n Entrevue tests \n\n";
  {
    BOOST_TEST(true);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FinDemonstrate)
{
  std::cout << "\n\n End Entrevue tests \n\n";
  {
    BOOST_TEST(true);
  }
}


BOOST_AUTO_TEST_SUITE_END() // Functions_tests
BOOST_AUTO_TEST_SUITE_END() // Cpp