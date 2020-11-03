//------------------------------------------------------------------------------
/// \file FoldExpression_tests.cpp
/// \ref C++17 - The Complete Guide: First Edition Paperback – September 6, 2019
/// by Nicolai M. Josuttis 
//------------------------------------------------------------------------------
#include "Cpp/Templates/FoldExpressions.h"

#include <boost/test/unit_test.hpp>
#include <string>

using Cpp::Templates::FoldExpressions::fold_sum;
using std::string;

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Templates)

BOOST_AUTO_TEST_SUITE(FoldExpressions_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TemplateParameterPackAcceptsZeroTemplateArguments)
{
  BOOST_TEST(fold_sum<>(11, -42.5) == -31.5);
  BOOST_TEST(fold_sum<>('a', string{"gtc"}) == "agtc");
}

//------------------------------------------------------------------------------
/// cf. Ch. 11 Fold Expressions, pp. 103, Josuttis (2019)
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FoldSumReturnsSumOfAllPassedArguments)
{
  BOOST_TEST(fold_sum(47, 11, -1) == 57);

  const double x {4269.9624};

  BOOST_TEST(fold_sum(47, 11, x, -1) == 4326.9624);

}

BOOST_AUTO_TEST_SUITE_END() // FoldExpressions_tests 

BOOST_AUTO_TEST_SUITE_END() // Templates
BOOST_AUTO_TEST_SUITE_END() // Cpp