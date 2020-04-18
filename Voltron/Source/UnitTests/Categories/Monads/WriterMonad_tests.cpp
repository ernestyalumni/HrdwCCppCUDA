//------------------------------------------------------------------------------
/// \file WriterMonad_tests.cpp
/// \ref Ivan Čukić, Functional Programming in C++,  Manning Publications;
/// 1st edition (November 19, 2018). ISBN-13: 978-1617293818
//------------------------------------------------------------------------------
#include "Categories/Monads/WriterMonad.h"

#include <boost/test/unit_test.hpp>
#include <string>

using Categories::Monads::WriterMonad::WriterMonadEndomorphism;
using Categories::Monads::WriterMonad::unit;
using Categories::Monads::WriterMonad::bind;

BOOST_AUTO_TEST_SUITE(Categories)
BOOST_AUTO_TEST_SUITE(Monads)
BOOST_AUTO_TEST_SUITE(WriterMonad_tests)

using LogEndomorphism = WriterMonadEndomorphism<int, std::string>;

// Test morphisms.

template <int N = 0>
WriterMonadEndomorphism<int, std::string> add_N(const int x)
{
  return WriterMonadEndomorphism<int, std::string>{
    x + N,
    "added " + std::to_string(N) + " "};
}

template <int N = 0>
WriterMonadEndomorphism<int, std::string> subtract_N(const int x)
{
  return WriterMonadEndomorphism<int, std::string>{
    x - N,
    "subtracted " + std::to_string(N) + " "};
}

BOOST_AUTO_TEST_SUITE(Construction_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CartesianProductExplicitlyConstructs)
{
  const WriterMonadEndomorphism<int, std::string> tx {42};
  BOOST_TEST(tx.value() == 42);
  BOOST_TEST(tx.log().empty());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CartesianProductConstructsWithCartesianProduct)
{
  const std::string start_log {"Start"};
  const WriterMonadEndomorphism<int, std::string> tx {42, start_log};
  BOOST_TEST(tx.value() == 42);
  BOOST_TEST((tx.log() == start_log));
}

BOOST_AUTO_TEST_SUITE_END() // Construction_tests

BOOST_AUTO_TEST_SUITE(Morphisms_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AddNIsAMorphism)
{
  const auto result = add_N<42>(69);
  BOOST_TEST(result.value() == 42 + 69);
  BOOST_TEST(result.log() == "added 42 ");
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SubtractNIsAMorphism)
{
  const auto result = subtract_N<42>(69);
  BOOST_TEST(result.value() == 69 - 42);
  BOOST_TEST(result.log() == "subtracted 42 ");
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(UnitActsAsAUnitAndIsVeryGeneral)
{
  const auto result = unit<int, LogEndomorphism>(1608);
  BOOST_TEST(result.value() == 1608);
  BOOST_TEST(result.log().empty());
}

BOOST_AUTO_TEST_SUITE_END() // Morphisms_tests

BOOST_AUTO_TEST_SUITE(Bind_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BindIsComposition)
{
  const LogEndomorphism previous_actions {42, "Started at 42,"};

  const auto result = bind(previous_actions, add_N<69>);

  BOOST_TEST(result.value() == 42 + 69);
  BOOST_TEST(result.log() == "Started at 42,added 69 ");

  const auto result2 = bind(result, subtract_N<10>);

  BOOST_TEST(result2.value() == 42 + 69 - 10);
  BOOST_TEST(result2.log() == "Started at 42,added 69 subtracted 10 ");
}

BOOST_AUTO_TEST_SUITE_END() // Morphisms_tests

BOOST_AUTO_TEST_SUITE_END() // WriterMonad_tests
BOOST_AUTO_TEST_SUITE_END() // Monads
BOOST_AUTO_TEST_SUITE_END() // Categories