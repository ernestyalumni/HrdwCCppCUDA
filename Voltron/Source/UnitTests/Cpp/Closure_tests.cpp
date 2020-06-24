//------------------------------------------------------------------------------
// \file Closure_tests.cpp
//------------------------------------------------------------------------------

#include <boost/test/unit_test.hpp>
#include <algorithm> // std::for_each
#include <functional>
#include <iostream>
#include <vector>

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Closure_tests)

// cf. https://nalaginrut.com/archives/2019/10/31/8%20essential%20patterns%20you%20should%20know%20about%20functional%20programming%20in%20c%2B%2B14
using fn_t = std::function<int(int)>;
fn_t func(int x)
{
	return [=](int y){ return x + y; };
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateClosure)
{
	{
		int y {5};

		auto add_2 = func(2);
		auto add_3 = func(3);
		y = add_2(y);
		BOOST_TEST(y == 7);
		y = add_3(y);
		BOOST_TEST(y == 10);
	}

	BOOST_TEST(true);
}

BOOST_AUTO_TEST_SUITE_END() // Closure_tests
BOOST_AUTO_TEST_SUITE_END() // Cpp