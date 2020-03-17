//------------------------------------------------------------------------------
/// \file Pointers_tests.cpp
/// \ref 
//------------------------------------------------------------------------------
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <memory>
#include <optional>

BOOST_AUTO_TEST_SUITE(Cpp) // The C++ Language
BOOST_AUTO_TEST_SUITE(Std)
BOOST_AUTO_TEST_SUITE(Optional_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateHasValue)
{
  std::optional<int> optional_1 {42};

  BOOST_TEST(optional_1.has_value());
}

BOOST_AUTO_TEST_SUITE_END() // Optional_tests
BOOST_AUTO_TEST_SUITE_END() // Std
BOOST_AUTO_TEST_SUITE_END() // Cpp