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

// cf. https://nalaginrut.com/archives/2019/10/31/8%20essential%20patterns%20you%20should%20know%20about%20functional%20programming%20in%20c%2B%2B14

BOOST_AUTO_TEST_SUITE(Lazy_tests)

// Thunk is a nullary function, say, a function without any parameters.
using thunk_t = std::function<int(void)>; 
// Return type is trivial, point is "nullary".
// Why does "nullary" matter? You have closure, so you can capture values you
// need, parameters are unnnecessary for us. May realize that thunk may help you
// to unify interface.

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateLazyPattern)
{
	{
		int x {5};
		auto thunk = 
			[x]()
				{
					std::cout << "now it run" << std::endl;
				};

		std::cout << "Thunk will not run before you run it" << std::endl;
		thunk();

		BOOST_TEST(true);
	}
}

template <typename T>
using UnaryF_t = std::function<T(void)>;

// https://stackoverflow.com/questions/265392/why-is-lazy-evaluation-useful
// https://bartoszmilewski.com/2014/04/21/getting-lazy-with-c/
// https://github.com/BartoszMilewski/Okasaki/blob/master/LazyQueue/Queue.h
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SquareAsLazyEvaluation)
{
	{
		double external_x {2};
		auto thunk = 
			[&external_x]()
			{
				return external_x * external_x;
			};

		external_x = thunk();
		external_x = thunk();
		external_x = thunk();
		BOOST_TEST(external_x == 256);
	}
}


BOOST_AUTO_TEST_SUITE_END() // Lazy_tests
BOOST_AUTO_TEST_SUITE_END() // Utilities