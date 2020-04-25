//------------------------------------------------------------------------------
/// \file StateMonad_tests.cpp
/// \author Ernest Yeung
//------------------------------------------------------------------------------
#include "Categories/Monads/StateMonad.h"

#include <array>
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <type_traits> // std::underlying_type
#include <utility> // std::pair

using Categories::Monads::StateMonad::AsLambdas::bind;
using Categories::Monads::StateMonad::AsLambdas::unit;
using Categories::Monads::StateMonad::Unit;
using Categories::Monads::StateMonad::compose;

BOOST_AUTO_TEST_SUITE(Categories)
BOOST_AUTO_TEST_SUITE(Monads)
BOOST_AUTO_TEST_SUITE(StateMonad_tests)

struct TestInputs
{
  double x_;
  bool yes_;
};

enum class TestEnumState: char
{
  a = 'a',
  b = 'b',
  c = 'c'
};

struct TestStateAsEnum
{
  TestEnumState state_;

  explicit TestStateAsEnum(const TestEnumState state):
    state_{state}
  {}

  explicit TestStateAsEnum():
    state_{TestEnumState::a}
  {}  
};

struct TestInputs2
{
  double value_;
  double operand_;
};

constexpr std::array<int, 2> two_states {{-1, 1}};

auto test_morphism_f(const TestInputs2& test_inputs) 
{
  return [&test_inputs](const int state)
  {
    double resulting_value;
    int resulting_state;
    if (state == two_states[1])
    {
      resulting_value = test_inputs.value_ + test_inputs.operand_;
    }
    else
    {
      resulting_value = test_inputs.value_ - test_inputs.operand_;
    }
    if (resulting_value < 0)
    {
      resulting_state = two_states[1];
    }
    else
    {
      resulting_state = two_states[0];
    }
    return unit(resulting_value)(resulting_state);
  };
}

class TestMorphismf
{
  public:

    TestMorphismf(const TestInputs2& test_inputs):
      inputs_{test_inputs}
    {}

    std::pair<int, double> operator()(const int state)
    {
      double resulting_value;
      int resulting_state;
      if (state == two_states[1])
      {
        resulting_value = inputs_.value_ + inputs_.operand_;
      }
      else
      {
        resulting_value = inputs_.value_ - inputs_.operand_;
      }
      if (resulting_value < 0)
      {
        resulting_state = two_states[1];
      }
      else
      {
        resulting_state = two_states[0];
      }
      return std::pair<int, double>(resulting_state, resulting_value);
    }

    TestInputs2 inputs() const
    {
      return inputs_;
    }

  private:

    TestInputs2 inputs_;
};

auto test_morphism_g(const double y) 
{
  return [y](const int state)
  {
    if (y > 0.0)
    {
      if (state == two_states[1])
      {
        return unit(-y)(two_states[0]);
      }
      else
      {
        return unit(y)(two_states[1]);
      }
    }
    else
    {
      return unit(y)(two_states[1]);
    }
  };
}

class TestMorphismg
{
  public:

    TestMorphismg(const double y):
      y_{y}
    {}

    auto operator()(const int state)
    {
      if (y_ > 0.0)
      {
        if (state == two_states[1])
        {
          Unit unit_y {-y_};
          return unit_y(two_states[0]);
        }
        else
        {

          Unit unit_y {y_};
          return unit_y(state);
        }
      }
      else
      {
        Unit unit_y {y_};
        return unit_y(state);
      }
    }

    double y() const
    {
      return y_;
    }

  private:

    double y_;
};

BOOST_AUTO_TEST_SUITE(Unit_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(UnitConstructs)
{
  {
    Unit<int> test_unit {5};
    BOOST_TEST(test_unit.input() == 5);
  }
  {
    int x {5};
    Unit<int> test_unit {x};
    BOOST_TEST(test_unit.input() == 5);
  }
  {
    // error: binding reference of type ‘int&’ to ‘const int’ discards qualifiers
    //constexpr int x {5};
    //Unit<int> test_unit {x};
    //BOOST_TEST(test_unit.input() == 5);
  }
  {
    Unit<TestInputs> test_unit {TestInputs{5.0, true}};
    BOOST_TEST(test_unit.input().x_ == 5.0);
    BOOST_TEST(test_unit.input().yes_ == true);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(UnitReturnsInternalHomFromStateToProductType)
{
  {
    Unit<int> test_unit {5};
    auto result = test_unit(TestEnumState::a);
    BOOST_TEST((result.first == TestEnumState::a));
    BOOST_TEST((result.second == 5));    
  }
  {
    const TestInputs test_inputs {5.0, false};
    Unit test_unit {test_inputs};
    auto result = test_unit(TestEnumState::b);
    BOOST_TEST((result.first == TestEnumState::b));
    BOOST_TEST(result.second.x_ == 5.0);
    BOOST_TEST(result.second.yes_ == false);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(UnitAsLambdaReturnsInternalHomFromStateToProductType)
{
  {
    auto result = unit(5)(TestEnumState::a);
    BOOST_TEST((result.first == TestEnumState::a));
    BOOST_TEST(result.second == 5);
  }
  {
    TestInputs test_inputs {42.0, true};
    TestEnumState state {TestEnumState::b};
    auto result = unit(test_inputs)(state);
    BOOST_TEST((result.first == TestEnumState::b));
    BOOST_TEST(result.second.x_ == 42.0);
    BOOST_TEST(result.second.yes_ == true);
  }
}

BOOST_AUTO_TEST_SUITE_END() // Unit_tests

BOOST_AUTO_TEST_SUITE(Morphisms)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TestMorphismsMaptoAnInternalHom)
{
  const TestInputs2 test_inputs {69.0, 42.0};
  int test_state {two_states[0]};
  {
    const auto result = test_morphism_f(test_inputs)(two_states[0]);
    BOOST_TEST(result.first == two_states[0]);
    BOOST_TEST(result.second == 27.0);
  }
  {
    auto morphism_f = TestMorphismf{test_inputs};
    auto result = morphism_f(test_state);
    BOOST_TEST(result.first == two_states[0]);
    BOOST_TEST(result.second == 27.0);
  }
  {
    const auto result = test_morphism_g(42.0)(two_states[1]);
    BOOST_TEST(result.first == two_states[0]);
    BOOST_TEST(result.second == -42.0);
  }
  {
    int state {two_states[1]};
    TestMorphismg g {42.0};
    auto result = g(state);
    //BOOST_TEST(result.first == two_states[0]);
    BOOST_TEST(result.second == -42.0);
  }
}

BOOST_AUTO_TEST_SUITE_END() // Morphisms

BOOST_AUTO_TEST_SUITE(BindTests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BindComposesMorphisms)
{
  {
    auto composed_morphisms = bind(test_morphism_g, test_morphism_f);
    const auto result =
      composed_morphisms(TestInputs2{69.0, 42.0})(two_states[1]);
    BOOST_TEST(result.first == two_states[1]);
    BOOST_TEST(result.second == 111.0);
  }
  {
    auto composed_morphisms = compose(test_morphism_g, test_morphism_f);
  }
}

BOOST_AUTO_TEST_SUITE_END() // BindTests

BOOST_AUTO_TEST_SUITE_END() // StateMonad_tests
BOOST_AUTO_TEST_SUITE_END() // Monads
BOOST_AUTO_TEST_SUITE_END() // Categories