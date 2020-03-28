//------------------------------------------------------------------------------
// \file Algorithms_tests.cpp
//------------------------------------------------------------------------------

#include <boost/test/unit_test.hpp>

#include <algorithm> // std::transform
#include <array>
#include <cctype> // std::toupper
#include <cstdio> // std::printf
#include <cstdlib> // std::atoi
#include <iostream>
#include <list>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(Algorithms_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StdForEachPrintsEachValue)
{
	{
		constexpr std::size_t N {100};

		// Initializer list doesn't work to make std::list have size of N.
		std::list<std::size_t> range_list (N);
		std::iota(range_list.begin(), range_list.end(), 0);

		std::size_t counter {0};		
		for (const auto& l_i : range_list)
		{
			BOOST_TEST_REQUIRE(l_i == counter);
			counter++;
		}
		BOOST_TEST_REQUIRE(counter == N);
		BOOST_TEST_REQUIRE(range_list.size() == N);

		// Initializer list just initializes to 2 values if done like this:
		// {N, '0'} or this {{N, '0'}}
		std::vector<unsigned char> test_values (N, '0');
		BOOST_TEST_REQUIRE(test_values.size() == N);

		std::transform(
			range_list.begin(),
			range_list.end(),
			test_values.begin(),
			[](const auto& l_i) -> unsigned char
			{
				return 'a' + l_i;
			});
		
	  std::for_each(
	  	range_list.begin(),
	    range_list.end(),
	    [&test_values](const auto& index)
	    {
	      std::printf("%01x ", test_values[index]);
	    });
	}
}

BOOST_AUTO_TEST_SUITE_END() // Algorithms_tests
BOOST_AUTO_TEST_SUITE_END() // Algorithms
BOOST_AUTO_TEST_SUITE_END() // Cpp

