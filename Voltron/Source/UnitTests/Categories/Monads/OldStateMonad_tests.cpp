//------------------------------------------------------------------------------
/// \file StateMonad_tests.cpp
/// \author Ernest Yeung
//------------------------------------------------------------------------------
#include "Categories/Monads/OldStateMonad.h"

#include <boost/test/unit_test.hpp>
#include <cmath>
#include <type_traits> // std::underlying_type
#include <utility> // std::pair

BOOST_AUTO_TEST_SUITE(Categories)
BOOST_AUTO_TEST_SUITE(Monads)
BOOST_AUTO_TEST_SUITE(OldStateMonad_tests)

using Categories::Monads::OldStateMonad::StateObject;
using Categories::Monads::OldStateMonad::StateObjectAsPair;
using Categories::Monads::OldStateMonad::multiplication_component;
using Categories::Monads::OldStateMonad::unit;

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

template <typename X, typename Function>
StateObject<X, TestEnumState> test_T_morphism(
  const StateObject<X, TestEnumState> tx, Function f)
{
  const auto y = f(tx.inputs());

  TestEnumState state;

  if (y == 0.0)
  {
    state = TestEnumState::a;
  }
  else if (y > 0.0)
  {
    state = TestEnumState::b;
  }
  else
  {
    state = TestEnumState::c;
  }

  return StateObject<X, TestEnumState>{y, state};
}

BOOST_AUTO_TEST_SUITE(CartesianProductDataType)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TestStatesDefaultConstructs)
{
  const TestEnumState enum_state {};

  BOOST_TEST(
    static_cast<std::underlying_type_t<TestEnumState>>(enum_state) == 0);

  TestStateAsEnum state {};
  BOOST_TEST(
    static_cast<std::underlying_type_t<TestEnumState>>(state.state_) == 'a');  
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TestEnumStateConstructs)
{
  TestEnumState enum_state {TestEnumState::b};

  BOOST_TEST(
    static_cast<std::underlying_type_t<TestEnumState>>(enum_state) == 'b');

  enum_state = TestEnumState::c;

  BOOST_TEST(
    static_cast<std::underlying_type_t<TestEnumState>>(enum_state) == 'c');

  TestStateAsEnum state {TestEnumState::b};
  BOOST_TEST(
    static_cast<std::underlying_type_t<TestEnumState>>(state.state_) == 'b');
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StateObjectConstructs)
{
  const StateObject<double, TestEnumState> state_object {
    42.0,
    TestEnumState::c};

  BOOST_TEST(state_object.inputs() == 42.0);
  BOOST_TEST((state_object.state() == TestEnumState::c));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StateObjectDefaultConstructs)
{
  {
    const StateObject<double, TestEnumState> state_object {42.0};

    BOOST_TEST(state_object.inputs() == 42.0);
    BOOST_TEST((static_cast<std::underlying_type_t<TestEnumState>>(
      state_object.state()) == 0));
  }
  {
    const StateObject<double, TestStateAsEnum> state_object {42.0};

    BOOST_TEST(state_object.inputs() == 42.0);
    BOOST_TEST((static_cast<std::underlying_type_t<TestEnumState>>(
      state_object.state().state_) == 'a'));
  }
}

BOOST_AUTO_TEST_SUITE_END() // CartesianProductDataType

BOOST_AUTO_TEST_SUITE(UnitComponent)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(UnitCanReturnStateObject)
{
  const StateObject<double, TestEnumState> tx {
    unit<double, TestEnumState>(6.022)};
  BOOST_TEST(tx.inputs() == 6.022);
  BOOST_TEST(
    (static_cast<std::underlying_type_t<TestEnumState>>(tx.state()) == 0));
}

BOOST_AUTO_TEST_SUITE_END() // UnitComponent

BOOST_AUTO_TEST_SUITE(MultiplicationComponent)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MultiplicationComponentReturnsStateObject)
{
  const std::pair<StateObject<double, TestEnumState>, TestEnumState> ttx {
    StateObject<double, TestEnumState>{6.674, TestEnumState::c},
    TestEnumState::b};

  const auto result = multiplication_component(ttx);

  BOOST_TEST(result.inputs() == 6.674);
  BOOST_TEST((
    static_cast<std::underlying_type_t<TestEnumState>>(result.state()) == 'c'));
}

BOOST_AUTO_TEST_SUITE_END() // MultiplicationComponent

BOOST_AUTO_TEST_SUITE(TMorphisms) 

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TestTMorphismTransitionsState)
{
  {
    const StateObject<double, TestEnumState> tx {0.0};
    const auto result = test_T_morphism(tx, sin);
    BOOST_TEST(result.inputs() == 0.0);
    BOOST_TEST((
      static_cast<std::underlying_type_t<TestEnumState>>(result.state()) ==
        'a'));
  }
  {
    const StateObject<double, TestEnumState> tx {0.707};
    const auto result = test_T_morphism(tx, sin);
    BOOST_TEST(result.inputs() == 0.64955575555642242);
    BOOST_TEST((
      static_cast<std::underlying_type_t<TestEnumState>>(result.state()) ==
        'b'));
  }
  {
    const StateObject<double, TestEnumState> tx {-0.107};
    const auto result = test_T_morphism(tx, sin);
    BOOST_TEST(result.inputs() == -0.10679594301412187);
    BOOST_TEST((
      static_cast<std::underlying_type_t<TestEnumState>>(result.state()) ==
        'c'));
  }
}

BOOST_AUTO_TEST_SUITE_END() // TMorphisms

BOOST_AUTO_TEST_SUITE_END() // StateMonad_tests
BOOST_AUTO_TEST_SUITE_END() // Monads
BOOST_AUTO_TEST_SUITE_END() // Categories