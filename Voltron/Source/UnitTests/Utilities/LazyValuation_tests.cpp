//------------------------------------------------------------------------------
// \file LazyValuation_tests.cpp
//------------------------------------------------------------------------------

#include <boost/test/unit_test.hpp>

#include "Utilities/LazyValuation.h"

using Utilities::LazyValuation;

BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(LazyValuation_tests)

// cf. http://pedromelendez.com/blog/2015/07/16/recursive-lambdas-in-c14/
// We cannot capture a variable declared using auto in its own initialization.
// lambdas in C++ are unique and unnamed; how to reference the function object
// cannot use this keyword inside body of lambda.
auto fibonacci = [](int x)
{
	auto implementation = [](int x, const auto& implementation) -> int
	{
		if (x == 0 || x == 1)
		{
			return 1;
		}
		else
		{
			return implementation(x - 1, implementation) +
				implementation(x - 2, implementation);
		}
	};

	return implementation(x, implementation);
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(LazyValuationConstructs)
{
	BOOST_TEST(fibonacci(0) == 1);
	BOOST_TEST(fibonacci(1) == 1);
	BOOST_TEST(fibonacci(2) == 2);
	BOOST_TEST(fibonacci(3) == 3);
	BOOST_TEST_REQUIRE(fibonacci(4) == 5);

	auto fibonacci_5 = []()
	{
		return fibonacci(5);
	};

	LazyValuation lazy_fibonacci_5_valuation {fibonacci_5};

	BOOST_TEST(lazy_fibonacci_5_valuation == 8);
}

// \ref https://gitlab.com/manning-fpcpp-book/code-examples/blob/master/chapter-06/lazy-val/main.cpp
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IsInvocableWorks)
{
	constexpr int number {6};

}

BOOST_AUTO_TEST_SUITE_END() // LazyValuation_tests
BOOST_AUTO_TEST_SUITE_END() // Utilities