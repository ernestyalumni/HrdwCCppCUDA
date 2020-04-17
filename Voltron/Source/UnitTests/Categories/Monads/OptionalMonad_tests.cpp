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