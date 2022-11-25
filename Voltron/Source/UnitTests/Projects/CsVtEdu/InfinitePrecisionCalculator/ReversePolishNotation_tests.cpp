#include "Projects/CsVtEdu/InfinitePrecisionCalculator/ReversePolishNotation.h"

#include <boost/test/unit_test.hpp>
#include <string>

template <typename T>
using ReversePolishNotation =
  CsVtEdu::InfinitePrecisionCalculator::ReversePolishNotation<T>;

using std::string;

BOOST_AUTO_TEST_SUITE(Projects)
BOOST_AUTO_TEST_SUITE(CsVtEdu)
BOOST_AUTO_TEST_SUITE(InfinitePrecisionCalculator)
BOOST_AUTO_TEST_SUITE(ReversePolishNotation_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
  ReversePolishNotation<int> rpn {};

  BOOST_TEST(true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ProcessInputPushesNumbers)
{
  const string operand_1 {"1234567890"};
  const string operand_2 {"987654321"};
  const string operand_3 {"42"};
  const string operand_4 {"69"};
  const string operand_5 {"2"};
  const string operand_6 {"3"};

  ReversePolishNotation<int> rpn {};

  rpn.process_input(operand_1);
  auto operand = rpn.top();

  BOOST_TEST(operand.size() == 10);
  BOOST_TEST(operand.head()->retrieve() == 0);
  BOOST_TEST(operand.head()->next()->retrieve() == 9);
  BOOST_TEST(operand.head()->next()->next()->retrieve() == 8);

  rpn.process_input(operand_2);
  operand = rpn.top();
  BOOST_TEST(operand.size() == 9);
  BOOST_TEST(operand.head()->retrieve() == 1);
  BOOST_TEST(operand.head()->next()->retrieve() == 2);
  BOOST_TEST(operand.head()->next()->next()->retrieve() == 3);

  rpn.process_input(operand_3);
  operand = rpn.top();
  BOOST_TEST(operand.size() == 2);
  BOOST_TEST(operand.head()->retrieve() == 2);
  BOOST_TEST(operand.head()->next()->retrieve() == 4);

  rpn.process_input(operand_4);
  operand = rpn.top();
  BOOST_TEST(operand.size() == 2);
  BOOST_TEST(operand.head()->retrieve() == 9);
  BOOST_TEST(operand.head()->next()->retrieve() == 6);

  rpn.process_input(operand_5);
  operand = rpn.top();
  BOOST_TEST(operand.size() == 1);
  BOOST_TEST(operand.head()->retrieve() == 2);

  rpn.process_input(operand_6);
  operand = rpn.top();
  BOOST_TEST(operand.size() == 1);
  BOOST_TEST(operand.head()->retrieve() == 3);\
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ProcessInputAddsNumbers)
{
  const string operand_1 {"42"};
  const string operand_2 {"69"};
  const string operator_1 {"+"};

  ReversePolishNotation<int> rpn {};
  rpn.process_input(operand_1);
  rpn.process_input(operand_2);
  rpn.process_input(operator_1);

  auto operand = rpn.top();
  BOOST_TEST(operand.size() == 3);
  BOOST_TEST(operand.head()->retrieve() == 1);
  BOOST_TEST(operand.head()->next()->retrieve() == 1);
  BOOST_TEST(operand.head()->next()->next()->retrieve() == 1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ProcessInputMultipliesNumbers)
{
  const string operand_1 {"42"};
  const string operand_2 {"69"};
  const string operator_1 {"*"};

  ReversePolishNotation<int> rpn {};
  rpn.process_input(operand_1);
  rpn.process_input(operand_2);
  rpn.process_input(operator_1);

  auto operand = rpn.top();
  BOOST_TEST(operand.size() == 4);
  BOOST_TEST(operand.head()->retrieve() == 8);
  BOOST_TEST(operand.head()->next()->retrieve() == 9);
  BOOST_TEST(operand.head()->next()->next()->retrieve() == 8);
}

// TODO: fix this.
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
/*
BOOST_AUTO_TEST_CASE(ProcessInputExponentiatesNumbers)
{
  const string operand_1 {"16"};
  const string operand_2 {"3"};
  const string operator_1 {"^"};

  ReversePolishNotation<int> rpn {};
  rpn.process_input(operand_1);
  rpn.process_input(operand_2);
  rpn.process_input(operator_1);

  auto operand = rpn.top();
  BOOST_TEST(operand.size() == 6);
  BOOST_TEST(operand.head()->retrieve() == 6);
  BOOST_TEST(operand.head()->next()->retrieve() == 9);
  BOOST_TEST(operand.head()->next()->next()->retrieve() == 0);
  BOOST_TEST(operand.head()->next()->next()->next()->retrieve() == 4);
}
*/

BOOST_AUTO_TEST_SUITE_END() // ReversePolishNotation_tests
BOOST_AUTO_TEST_SUITE_END() // InfinitePrecisionCalculator
BOOST_AUTO_TEST_SUITE_END() // CsVtEdu
BOOST_AUTO_TEST_SUITE_END() // Projects