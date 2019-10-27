//------------------------------------------------------------------------------
// \file OnlyOneArgument_test.cpp
//------------------------------------------------------------------------------
#include "Utilities/OnlyOneArgument.h"

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <type_traits>

using Utilities::OnlyOneArgument;

BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(OnlyOneArgument_tests)

int dummy_f0()
{ return 42; }

void dummy_f0b()
{ return; }

int dummy_f1(const int x)
{ return x; }

void dummy_f1b(const int x)
{ 
	std::cout << x << '\n';	
	return;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(OnlyOneArgumentDistinguishesFunctionsByArgumentNumber)
{
//	OnlyOneArgument<decltype(dummy_f0)> x;
	//BOOST_TEST((OnlyOneArgument<decltype(dummy_f0)>::value));

	//bool(OnlyOneArgument<decltype(dummy_f0)>);

	long value = 1L;

	BOOST_TEST(
		dummy_f1(static_cast<OnlyOneArgumentT<decltype(dummy_f1b)>>(value)) == 1);

	BOOST_TEST(
		(std::is_invocable_v<
			decltype(dummy_f1),
			OnlyOneArgumentT<decltype(dummy_f1b)>
			>));

	BOOST_TEST(
		(std::is_invocable_v<
			decltype(dummy_f1b),
			OnlyOneArgumentT<decltype(dummy_f1b)>
			>));

	BOOST_TEST(
		!(std::is_invocable_v<
			decltype(dummy_f0),
			OnlyOneArgumentT<decltype(dummy_f1b)>
			>));

	BOOST_TEST(
		!(std::is_invocable_v<
			decltype(dummy_f0b),
			OnlyOneArgumentT<decltype(dummy_f1b)>
			>));

}


BOOST_AUTO_TEST_SUITE_END() // OnlyOneArgument_tests
BOOST_AUTO_TEST_SUITE_END() // Utilities
