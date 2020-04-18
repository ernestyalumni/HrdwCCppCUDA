//------------------------------------------------------------------------------
/// \file WriterMonad_tests.cpp
/// \ref Ivan Čukić, Functional Programming in C++,  Manning Publications;
/// 1st edition (November 19, 2018). ISBN-13: 978-1617293818
//------------------------------------------------------------------------------
#include "Categories/Monads/ReaderMonad.h"

#include <boost/test/unit_test.hpp>
#include <string>

using Categories::Monads::ReaderMonad::ask;
using Categories::Monads::ReaderMonad::return_;
using Categories::Monads::ReaderMonad::unit;

BOOST_AUTO_TEST_SUITE(Categories)
BOOST_AUTO_TEST_SUITE(Monads)
BOOST_AUTO_TEST_SUITE(ReaderMonad_tests)

struct TestEnvironment
{
  bool yes_;
  double x_;
  std::string s_;
};

TestEnvironment initial_test_environment {false, 42.0, "Start"};

// Test morphisms.

double r(const TestEnvironment& environment)
{
  if (environment.yes_)
  {
    return environment.x_;
  }
  return 0.0;
}

std::string s(const TestEnvironment& environment)
{
  if (environment.yes_)
  {
    return environment.s_;
  }
  return "Go back.";
}

BOOST_AUTO_TEST_SUITE(Morphisms_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TestMorphismsMapEnvironmentToCategoryObjectType)
{
  const TestEnvironment test_environment {true, 1.616, "Middle"};

  BOOST_TEST(r(initial_test_environment) == 0.0);
  BOOST_TEST(r(test_environment) == 1.616);
  BOOST_TEST(s(initial_test_environment) == "Go back.");
  BOOST_TEST(s(test_environment) == "Middle");  
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(UnitReturnsConstantFunctions)
{
  {
    auto unit_component = unit<double, TestEnvironment>(69.0);
    BOOST_TEST(unit_component(initial_test_environment) == 69.0);
  }
  {
    auto unit_component = unit<double, TestEnvironment>(1.616);
    BOOST_TEST(unit_component(initial_test_environment) == 1.616);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(LambdaVersionOfUnitReturnsConstantFunctions)
{
  const auto unit_component = return_(10.0);
  BOOST_TEST(unit_component(initial_test_environment) == 10.0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AskReturnsEnvironmentAsIdentity)
{
  const auto result = ask(initial_test_environment);
  BOOST_TEST(!result.yes_);
  BOOST_TEST(result.x_ == 42.0);
  BOOST_TEST(result.s_ == "Start");
}


BOOST_AUTO_TEST_SUITE_END() // Morphisms_tests

BOOST_AUTO_TEST_SUITE_END() // ReaderMonad_tests
BOOST_AUTO_TEST_SUITE_END() // Monads
BOOST_AUTO_TEST_SUITE_END() // Categories