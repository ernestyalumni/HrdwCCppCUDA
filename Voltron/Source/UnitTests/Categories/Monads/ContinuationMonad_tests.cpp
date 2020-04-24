//------------------------------------------------------------------------------
/// \file ContinuationMonad_tests.cpp
/// \author Ernest Yeung
//------------------------------------------------------------------------------
//#include "Categories/Monads/ContinuationMonad.h"

#define BOOST_THREAD_PROVIDES_FUTURE
#define BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION

#include "Categories/Monads/ContinuationMonad.h"

#include <boost/test/unit_test.hpp>
#include <future>
#include <optional>
#include <string>
#include <utility>

using Categories::Monads::ContinuationMonad::AsLambdas::return_;
using Categories::Monads::ContinuationMonad::AsLambdas::runContinuation;
using Categories::Monads::ContinuationMonad::apply_endomorphism;
using Categories::Monads::ContinuationMonad::unit;

using namespace Categories::Monads::ContinuationMonad;

BOOST_AUTO_TEST_SUITE(Categories)
BOOST_AUTO_TEST_SUITE(Monads)
BOOST_AUTO_TEST_SUITE(ContinuationMonad_tests)

// Test morphisms.

std::optional<std::string> user_full_name(const std::string name)
{
  if (name.empty() || name == "None")
  {
    return {};
  }
  else
  {
    return std::make_optional("Username: " + name);
  }
}

std::optional<std::string> to_html(const std::string name)
{
  if (name.empty() || name == "NA")
  {
    return {};
  }
  else
  {
    return std::make_optional("http://" + name);
  }
}

std::pair<bool, std::string> h(const std::optional<std::string> text)
{
  if (!text)
  {
    return std::make_pair(false, "");
  }
  if (text.value() == "What you want.")
  {
    return std::make_pair(true, text.value());
  }
  else
  {
    return std::make_pair(false, text.value());
  }
}

template <int Divisor>
std::pair<bool, int> integer_division(const int input)
{
  if (Divisor == 0)
  {
    return std::make_pair(false, input);
  }
  else
  {
    return std::make_pair(true, input / Divisor);
  }
}

BOOST_AUTO_TEST_SUITE(Endomorphisms)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ApplyEndomorphismAppliesInternalHomToInput)
{
  {
    const auto result = apply_endomorphism(user_full_name, "Sainte-Foy");
    BOOST_TEST(result.value() == "Username: Sainte-Foy");
  }
  {
    const auto result = apply_endomorphism(to_html, "Sainte-Foy");
    BOOST_TEST(result.value() == "http://Sainte-Foy");
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(RunContinuationAppliesInternalHomToInput)
{
  {
    const auto result = runContinuation(user_full_name, "Sainte-Foy");
    BOOST_TEST(result.value() == "Username: Sainte-Foy");
  }
  {
    const auto result = runContinuation(to_html, "Sainte-Foy");
    BOOST_TEST(result.value() == "http://Sainte-Foy");
  }
}

BOOST_AUTO_TEST_SUITE_END() // Endomorphisms

BOOST_AUTO_TEST_SUITE(Return)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TestMorphismsWork)
{
  {
    auto result = user_full_name("Sainte-Foy");
    BOOST_TEST(result.value() == "Username: Sainte-Foy");
  }
  {
    const auto result1 = integer_division<2>(5);
    BOOST_TEST(result1.first);
    BOOST_TEST(result1.second == 2);    
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ReturnReturnsEndomorphismObject)
{
  auto returned_unit = return_("Sainte-Foy");
  {
    const auto result = returned_unit(user_full_name);
    BOOST_TEST(result.value() == "Username: Sainte-Foy");
  }
  {
    const auto result = returned_unit(to_html);
    BOOST_TEST(result.value() == "http://Sainte-Foy");
  }
  auto returned_unit1 = return_(10);
  {
  }
}

BOOST_AUTO_TEST_SUITE_END() // Return

// cf. https://github.com/dbrattli/OSlash/blob/master/tests/test_cont.py

BOOST_AUTO_TEST_SUITE(UnitComponent)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(
  UnitComponentReturnsInternalHomMappingInternalHomToReturnType)
{
  auto returned_endomorphism = unit<std::string>("Helene Desportes");
  {
    const auto result = returned_endomorphism(user_full_name);
    BOOST_TEST(result.value() == "Username: Helene Desportes");
  }
  {
    const auto result = returned_endomorphism(to_html);
    BOOST_TEST(result.value() == "http://Helene Desportes");
  }
}

// cf. https://github.com/dbrattli/OSlash/blob/master/tests/test_cont.py
// The following reproduces the unit tests in the referenced link.

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(UnitComponentDelaysComputation)
{
  auto test_internal_hom = [](const int value)
  {
    return std::to_string(value);
  };
  {
    auto addition = [](const auto x, const auto y)
    {
      return unit(x + y);
    };
    auto square = [](const auto x)
    {
      return unit(x * x);
    };
    // Results is of type [[X, Y], Y]
    auto delay_addition = addition(25, 9);
    auto delay_square = square(5);

    const auto addition_result = delay_addition(test_internal_hom);
    const auto square_result = delay_square(test_internal_hom);

    BOOST_TEST_REQUIRE(addition_result == "34");
    BOOST_TEST_REQUIRE(square_result == "25");

    {
      auto create_add_x = [](const int x)
      {
        return [x](const auto y)
        {
          return (x + y);
        };
      };
      
    }

  }
  {
    auto addition = [](const auto x, const auto y)
    {
      return return_(x + y);
    };
    auto square = [](const auto x)
    {
      return return_(x * x);
    };
    auto delay_addition = addition(25, 9);
    auto delay_square = square(5);

    const auto addition_result = delay_addition(test_internal_hom);
    const auto square_result = delay_square(test_internal_hom);

    BOOST_TEST_REQUIRE(addition_result == "34");
    BOOST_TEST_REQUIRE(square_result == "25");
  }

}

BOOST_AUTO_TEST_SUITE_END() // UnitComponent

BOOST_AUTO_TEST_SUITE(Bind)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BindAsLambdas)
{
  auto composed_fg = AsLambdas::bind(to_html, user_full_name);
//  composed_fg(h);

}

BOOST_AUTO_TEST_SUITE_END() // Bind


BOOST_AUTO_TEST_SUITE_END() // ContinuationMonad_tests
BOOST_AUTO_TEST_SUITE_END() // Monads
BOOST_AUTO_TEST_SUITE_END() // Categories