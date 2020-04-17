//------------------------------------------------------------------------------
/// \file PointersArraysReferences_tests.cpp
/// \ref Ivan Čukić, Functional Programming in C++,  Manning Publications;
/// 1st edition (November 19, 2018). ISBN-13: 978-1617293818
//------------------------------------------------------------------------------
#include "Categories/Monads/OptionalMonad.h"

#include <boost/test/unit_test.hpp>
#include <optional>
#include <string>

using Categories::Monads::OptionalMonad::endomorphism_morphism_map;
using Categories::Monads::OptionalMonad::multiplication_component;

BOOST_AUTO_TEST_SUITE(Categories)
BOOST_AUTO_TEST_SUITE(Monads)
BOOST_AUTO_TEST_SUITE(OptionalMonad_tests)

// cf. Čukić (2018), Ch. 10

std::optional<std::string> user_full_name(const std::string& login)
{
  return {};
}

std::optional<std::string> to_html(const std::string& text)
{
  return {};
}

std::optional<std::string> usable_user_full_name(const std::string& login)
{
  if (login == "None")
  {
    return {};
  }

  return std::make_optional<std::string>(login);
}

std::optional<std::string> usable_to_html(const std::string& text)
{
  if (text == "No address")
  {
    return {};
  }

  return std::make_optional<std::string>(text);
}

// Testfunctions act as morphisms f : X \to T(Y) where T is the endomorphism.
// T : X \to T(X), where T(X) is of type std::optional
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TestFunctionBehavesAsAMorphism)
{
  const std::string test_name {"Jean-Francois Roberval"};
  // result of morphism f of type T(Y), as f : X \to T(Y)
  const auto result = usable_user_full_name(test_name);
  BOOST_TEST(static_cast<bool>(result));
  BOOST_TEST((result.value() == "Jean-Francois Roberval"));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(EndomorphismMorphismMapWorksOnFunctionReturningStdOptional)
{
  const auto test_full_name =
    std::make_optional<std::string>("Jacques Cartier");

  const auto result =
    endomorphism_morphism_map(test_full_name, usable_user_full_name);

  BOOST_TEST(static_cast<bool>(result));
  BOOST_TEST((result.value() == test_full_name.value()));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(UseMainFromOptionalMonadExampleFromCukic)
{
  std::optional<std::string> login;

  multiplication_component(
    endomorphism_morphism_map(
      multiplication_component(
        endomorphism_morphism_map(
          login,
          user_full_name)),
      to_html));

  auto login_and_user_full_name =
    multiplication_component(
      endomorphism_morphism_map(
        multiplication_component(
          endomorphism_morphism_map(
            login,
            user_full_name)),
        to_html)); 

  BOOST_TEST(true);
}

BOOST_AUTO_TEST_SUITE_END() // OptionalMonad_tests
BOOST_AUTO_TEST_SUITE_END() // Monads
BOOST_AUTO_TEST_SUITE_END() // Categories