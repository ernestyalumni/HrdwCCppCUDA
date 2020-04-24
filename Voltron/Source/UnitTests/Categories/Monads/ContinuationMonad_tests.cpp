//------------------------------------------------------------------------------
/// \file ContinuationMonad_tests.cpp
/// \author Ernest Yeung
//------------------------------------------------------------------------------
//#include "Categories/Monads/ContinuationMonad.h"

#define BOOST_THREAD_PROVIDES_FUTURE
#define BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION

#include "Categories/Monads/ContinuationMonad.h"

#include <boost/test/unit_test.hpp>
#include <cmath>
#include <future>
#include <optional>
#include <string>
#include <utility>

using Categories::Monads::ContinuationMonad::AsLambdas::eval;
using Categories::Monads::ContinuationMonad::AsLambdas::return_;
using Categories::Monads::ContinuationMonad::AsLambdas::runContinuation;
using Categories::Monads::ContinuationMonad::apply_endomorphism;
using Categories::Monads::ContinuationMonad::evaluate;
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
    BOOST_TEST(addition(25, 9)(evaluate<int>) == 34);
    {
      auto create_add_x = [](const int x)
      {
        return [x](const auto y)
        {
          return (x + y);
        };
      };
      auto add_6 = create_add_x(6);
      auto add_7 = create_add_x(7);

      BOOST_TEST(delay_square(add_6) == 31);
      BOOST_TEST(delay_square(add_7) == 32);
      BOOST_TEST(square(5)(add_6) == 31);
      BOOST_TEST(square(5)(add_7) == 32);

      auto intermediate = square(5)(
        [addition](const int xx)
        {
          return addition(xx, 8);
        });

      BOOST_TEST_REQUIRE(intermediate(evaluate<int>) == 33);

      auto intermediate_f = [square, addition](const int x, const int y)
      {
        return square(x)(
          [addition, y](const int xx)
          {
            return addition(xx, y);
          });
      };

      auto intermediate_f_result = intermediate_f(3, 6)(evaluate<int>);
      BOOST_TEST(intermediate_f_result == 15);

      auto pythagoras_formula = [addition, square](const int x, const int y)
      {
        return square(x)(
          [addition, square, y](const int xx)
          {
            return square(y)(
              [addition, xx](const int yy)
              {
                return addition(xx, yy);
              });
          });
      };
      BOOST_TEST(pythagoras_formula(5, 6)(evaluate<int>) == 61);
      BOOST_TEST(pythagoras_formula(4, 4)(evaluate<int>) == 32);
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

    {
      auto pythagoras_formula = [addition, square](const int x, const int y)
      {
        return square(x)(
          [addition, square, y](const int xx)
          {
            return square(y)(
              [addition, xx](const int yy)
              {
                return addition(xx, yy);
              });
          });
      };
      BOOST_TEST(pythagoras_formula(5, 6)(evaluate<int>) == 61);
      BOOST_TEST(pythagoras_formula(4, 4)(evaluate<int>) == 32);
    }
  }
}

int pair_add(const std::pair<int, int> inputs)
{
  return inputs.first + inputs.second;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(UnitComponentTreatsFunctionsAsFirstClassValues)
{
  {
    auto pair_addition = unit(pair_add);
    BOOST_TEST_REQUIRE(pair_addition(eval)(std::make_pair<int, int>(40, 2)));
    BOOST_TEST(unit(pair_add)(eval)(std::make_pair<int, int>(40, 2)));
  }
  {
    auto pair_addition = return_(pair_add);
    BOOST_TEST_REQUIRE(pair_addition(eval)(std::make_pair<int, int>(40, 2)));
    BOOST_TEST(return_(pair_add)(eval)(std::make_pair<int, int>(40, 2)));    
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SimpleUnitComponentExamples)
{
  auto morphism_c = [](const auto x) -> std::string
  {
    return "Done: " + std::to_string(x);
  };
  {
    auto f = [](const auto x)
    {
      return unit(std::pow(x, 3));
    };
    auto g = [](const auto x)
    {
      return unit(x - 2);
    };
    auto h = [f, g](const auto x)
    {
      return (x == 5) ? f(x) : g(x);
    };
    auto do_c = unit(4.0)(h);
    BOOST_TEST_REQUIRE(do_c(evaluate<float>) == 2.0);
    BOOST_TEST(do_c(morphism_c) == "Done: 2.000000");
  }
  {
    auto f = [](const auto x)
    {
      return return_(std::pow(x, 3));
    };
    auto g = [](const auto x)
    {
      return return_(x - 2);
    };
    auto h = [f, g](const auto x)
    {
      return (x == 5) ? f(x) : g(x);
    };
    auto do_c = return_(4.0)(h);
    BOOST_TEST_REQUIRE(do_c(evaluate<float>) == 2.0);
    BOOST_TEST(do_c(morphism_c) == "Done: 2.000000");
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