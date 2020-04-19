//------------------------------------------------------------------------------
/// \file StateMonad_tests.cpp
/// \ref Ivan Čukić, Functional Programming in C++,  Manning Publications;
/// 1st edition (November 19, 2018). ISBN-13: 978-1617293818
//------------------------------------------------------------------------------
#include "Categories/Monads/ReaderMonad.h"

#include <boost/test/unit_test.hpp>
#include <regex>
#include <string>

using Categories::Monads::ReaderMonad::apply_morphism;
using Categories::Monads::ReaderMonad::ask;
using Categories::Monads::ReaderMonad::bind;
using Categories::Monads::ReaderMonad::bind_;
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
const TestEnvironment test_environment {true, 1.616, "Middle"};

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

// test_morphisms belongs to Hom(E, X).
// Example from https://nbviewer.jupyter.org/github/dbrattli/OSlash/blob/master/notebooks/Reader.ipynb

// Auxiliary function to test morphism.
std::string transform_text(const std::string text)
{
  return std::regex_replace(text, std::regex("Hi"), "Hello");
}

// Serves as a morphism in (E \to X)
auto test_string_morphism = [](const std::string& name) -> std::string
{
  return "Hi " + name + "!";
};

// Serves as a morphism f : X \to T(Y) that will have an endomorphism applied to
// it.
auto another_string_morphism(const std::string& text)
{
  const std::string replaced_text {transform_text(text)};

  return unit<std::string, std::string>(replaced_text);
}

// Modified example.
auto test_double_string_morphism = [](const double value) -> std::string
{
  return "Value : " + std::to_string(value) + ", ";
};

BOOST_AUTO_TEST_SUITE(Morphisms_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TestMorphismsMapEnvironmentToCategoryObjectType)
{
  BOOST_TEST(r(initial_test_environment) == 0.0);
  BOOST_TEST(r(test_environment) == 1.616);
  BOOST_TEST(s(initial_test_environment) == "Go back.");
  BOOST_TEST(s(test_environment) == "Middle");  
  BOOST_TEST(test_string_morphism("Jean Talon") == "Hi Jean Talon!");
  BOOST_TEST(test_double_string_morphism(2.998) == "Value : 2.998000, ");
  BOOST_TEST(transform_text("Hi there") == "Hello there");
  BOOST_TEST(another_string_morphism("Hi Hi Hi you")("42") ==
    "Hello Hello Hello you");
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
  // Example from https://nbviewer.jupyter.org/github/dbrattli/OSlash/blob/master/notebooks/Reader.ipynb
  {
    auto unit_component = unit<int, std::string>(42);
    BOOST_TEST(unit_component("Ignored") == 42);
  }
  {
    auto unit_component = unit<std::string, std::string>("Hello there");
    BOOST_TEST(unit_component("Bonjour") == "Hello there");
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

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ApplyMorphismReturnsValueOfTypeX)
{
  {
    const auto result = apply_morphism(r, initial_test_environment);
    BOOST_TEST(result == 0.0);
  }
  {
    const auto result = apply_morphism(r, test_environment);
    BOOST_TEST(result == 1.616);
  }
}

BOOST_AUTO_TEST_SUITE_END() // Morphisms_tests

BOOST_AUTO_TEST_SUITE(Bind_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TestingStepsInBind)
{
  const std::string test_environment_string {"Sieur de la Salle"};
  {
    auto f_on_X =
      another_string_morphism(test_string_morphism(test_environment_string));
    BOOST_TEST(f_on_X(test_environment_string) == "Hello Sieur de la Salle!");
  }
  {
    auto f_on_X = another_string_morphism(
      apply_morphism(test_string_morphism, test_environment_string));

    BOOST_TEST(apply_morphism(f_on_X, test_environment_string) ==
      "Hello Sieur de la Salle!");
  }
}

// TODO: Make bind work with templates.
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BindWorksOnStringMorphisms)
{
  auto result = bind(test_string_morphism, another_string_morphism);
  //result("Le Royer");
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BindAsLambdaWorksOnStringMorphisms)
{
  auto result = bind_(test_string_morphism, another_string_morphism);

  BOOST_TEST(result("Le Royer") == "Hello Le Royer!");
}

BOOST_AUTO_TEST_SUITE_END() // Bind_tests

BOOST_AUTO_TEST_SUITE_END() // ReaderMonad_tests
BOOST_AUTO_TEST_SUITE_END() // Monads
BOOST_AUTO_TEST_SUITE_END() // Categories