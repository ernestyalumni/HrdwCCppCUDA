//------------------------------------------------------------------------------
// \file Algorithms_tests.cpp
//------------------------------------------------------------------------------
#include <algorithm> // std::transform
#include <array>
#include <boost/test/unit_test.hpp>
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
		
		//std::array<char, N> buffer {};
		//char buffer[N];
		std::array<unsigned char, N> buffer {};

	  std::for_each(
	  	range_list.begin(),
	    range_list.end(),
	    [&test_values, &buffer](const auto& index)
	    {
	    	buffer[index] = test_values[index];
	    	//std::snprintf(buffer + index, 1, "%01x", test_values[index]);
	    	//std::snprintf(&(buffer.data()[index]), 1, "%01x", test_values[index]);
	      //std::printf("%01x ", test_values[index]);
	    });

	  BOOST_TEST(buffer[0] == 0x61);
	  BOOST_TEST(buffer[1] == 0x62);
	  BOOST_TEST(buffer[2] == 0x63);
	  BOOST_TEST(buffer[3] == 0x64);
	}
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StdNextPermutationsGetsAllPermutations)
{
	constexpr std::size_t N {3};
	std::array<unsigned int, N> a {};
	std::iota(a.begin(), a.end(), 0);
	BOOST_TEST(a[0] == 0);
	BOOST_TEST(a[1] == 1);
	BOOST_TEST(a[2] == 2);

	BOOST_TEST(std::next_permutation(a.begin(), a.end()));

	BOOST_TEST(a[0] == 0);
	BOOST_TEST(a[1] == 2);
	BOOST_TEST(a[2] == 1);

	BOOST_TEST(std::next_permutation(a.begin(), a.end()));

	BOOST_TEST(a[0] == 1);
	BOOST_TEST(a[1] == 0);
	BOOST_TEST(a[2] == 2);

	BOOST_TEST(std::next_permutation(a.begin(), a.end()));

	BOOST_TEST(a[0] == 1);
	BOOST_TEST(a[1] == 2);
	BOOST_TEST(a[2] == 0);

	BOOST_TEST(std::next_permutation(a.begin(), a.end()));

	BOOST_TEST(a[0] == 2);
	BOOST_TEST(a[1] == 0);
	BOOST_TEST(a[2] == 1);

	BOOST_TEST(std::next_permutation(a.begin(), a.end()));

	BOOST_TEST(a[0] == 2);
	BOOST_TEST(a[1] == 1);
	BOOST_TEST(a[2] == 0);

	BOOST_TEST(!std::next_permutation(a.begin(), a.end()));

	BOOST_TEST(a[0] == 0);
	BOOST_TEST(a[1] == 1);
	BOOST_TEST(a[2] == 2);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StdNextPermutationsGetsAllPermutationsForSize4)
{
	constexpr std::size_t N {4};
	std::array<unsigned int, N> a {};
	std::iota(a.begin(), a.end(), 0);
	BOOST_TEST(a[0] == 0);
	BOOST_TEST(a[1] == 1);
	BOOST_TEST(a[2] == 2);
	BOOST_TEST(a[3] == 3);

	BOOST_TEST(std::next_permutation(a.begin(), a.end()));

	BOOST_TEST(a[0] == 0);
	BOOST_TEST(a[1] == 1);
	BOOST_TEST(a[2] == 3);
	BOOST_TEST(a[3] == 2);

	BOOST_TEST(std::next_permutation(a.begin(), a.end()));

	BOOST_TEST(a[0] == 0);
	BOOST_TEST(a[1] == 2);
	BOOST_TEST(a[2] == 1);
	BOOST_TEST(a[3] == 3);

	BOOST_TEST(std::next_permutation(a.begin(), a.end()));

	BOOST_TEST(a[0] == 0);
	BOOST_TEST(a[1] == 2);
	BOOST_TEST(a[2] == 3);
	BOOST_TEST(a[3] == 1);

	BOOST_TEST(std::next_permutation(a.begin(), a.end()));

	BOOST_TEST(a[0] == 0);
	BOOST_TEST(a[1] == 3);
	BOOST_TEST(a[2] == 1);
	BOOST_TEST(a[3] == 2);

	BOOST_TEST(std::next_permutation(a.begin(), a.end()));

	BOOST_TEST(a[0] == 0);
	BOOST_TEST(a[1] == 3);
	BOOST_TEST(a[2] == 2);
	BOOST_TEST(a[3] == 1);

	BOOST_TEST(std::next_permutation(a.begin(), a.end()));

	BOOST_TEST(a[0] == 1);
	BOOST_TEST(a[1] == 0);
	BOOST_TEST(a[2] == 2);
	BOOST_TEST(a[3] == 3);

	BOOST_TEST(std::next_permutation(a.begin(), a.end()));

	BOOST_TEST(a[0] == 1);
	BOOST_TEST(a[1] == 0);
	BOOST_TEST(a[2] == 3);
	BOOST_TEST(a[3] == 2);

	BOOST_TEST(std::next_permutation(a.begin(), a.end()));

	BOOST_TEST(a[0] == 1);
	BOOST_TEST(a[1] == 2);
	BOOST_TEST(a[2] == 0);
	BOOST_TEST(a[3] == 3);

	BOOST_TEST(std::next_permutation(a.begin(), a.end()));

	BOOST_TEST(a[0] == 1);
	BOOST_TEST(a[1] == 2);
	BOOST_TEST(a[2] == 3);
	BOOST_TEST(a[3] == 0);

	BOOST_TEST(std::next_permutation(a.begin(), a.end()));

	BOOST_TEST(a[0] == 1);
	BOOST_TEST(a[1] == 3);
	BOOST_TEST(a[2] == 0);
	BOOST_TEST(a[3] == 2);

	BOOST_TEST(std::next_permutation(a.begin(), a.end()));

	BOOST_TEST(a[0] == 1);
	BOOST_TEST(a[1] == 3);
	BOOST_TEST(a[2] == 2);
	BOOST_TEST(a[3] == 0);

	BOOST_TEST(std::next_permutation(a.begin(), a.end()));

	BOOST_TEST(a[0] == 2);
	BOOST_TEST(a[1] == 0);
	BOOST_TEST(a[2] == 1);
	BOOST_TEST(a[3] == 3);

	BOOST_TEST(std::next_permutation(a.begin(), a.end()));

	BOOST_TEST(a[0] == 2);
	BOOST_TEST(a[1] == 0);
	BOOST_TEST(a[2] == 3);
	BOOST_TEST(a[3] == 1);

	BOOST_TEST(std::next_permutation(a.begin(), a.end()));

	BOOST_TEST(a[0] == 2);
	BOOST_TEST(a[1] == 1);
	BOOST_TEST(a[2] == 0);
	BOOST_TEST(a[3] == 3);

	BOOST_TEST(std::next_permutation(a.begin(), a.end()));

	BOOST_TEST(a[0] == 2);
	BOOST_TEST(a[1] == 1);
	BOOST_TEST(a[2] == 3);
	BOOST_TEST(a[3] == 0);

	BOOST_TEST(std::next_permutation(a.begin(), a.end()));

	BOOST_TEST(a[0] == 2);
	BOOST_TEST(a[1] == 3);
	BOOST_TEST(a[2] == 0);
	BOOST_TEST(a[3] == 1);

	BOOST_TEST(std::next_permutation(a.begin(), a.end()));

	BOOST_TEST(a[0] == 2);
	BOOST_TEST(a[1] == 3);
	BOOST_TEST(a[2] == 1);
	BOOST_TEST(a[3] == 0);

	BOOST_TEST(std::next_permutation(a.begin(), a.end()));

	BOOST_TEST(a[0] == 3);
	BOOST_TEST(a[1] == 0);
	BOOST_TEST(a[2] == 1);
	BOOST_TEST(a[3] == 2);

	BOOST_TEST(std::next_permutation(a.begin(), a.end()));

	BOOST_TEST(a[0] == 3);
	BOOST_TEST(a[1] == 0);
	BOOST_TEST(a[2] == 2);
	BOOST_TEST(a[3] == 1);

	BOOST_TEST(std::next_permutation(a.begin(), a.end()));

	BOOST_TEST(a[0] == 3);
	BOOST_TEST(a[1] == 1);
	BOOST_TEST(a[2] == 0);
	BOOST_TEST(a[3] == 2);

	BOOST_TEST(std::next_permutation(a.begin(), a.end()));

	BOOST_TEST(a[0] == 3);
	BOOST_TEST(a[1] == 1);
	BOOST_TEST(a[2] == 2);
	BOOST_TEST(a[3] == 0);

	BOOST_TEST(std::next_permutation(a.begin(), a.end()));

	BOOST_TEST(a[0] == 3);
	BOOST_TEST(a[1] == 2);
	BOOST_TEST(a[2] == 0);
	BOOST_TEST(a[3] == 1);

	BOOST_TEST(std::next_permutation(a.begin(), a.end()));

	BOOST_TEST(a[0] == 3);
	BOOST_TEST(a[1] == 2);
	BOOST_TEST(a[2] == 1);
	BOOST_TEST(a[3] == 0);

	BOOST_TEST(!std::next_permutation(a.begin(), a.end()));

	BOOST_TEST(a[0] == 0);
	BOOST_TEST(a[1] == 1);
	BOOST_TEST(a[2] == 2);
	BOOST_TEST(a[3] == 3);
}

BOOST_AUTO_TEST_SUITE_END() // Algorithms_tests
BOOST_AUTO_TEST_SUITE_END() // Algorithms
BOOST_AUTO_TEST_SUITE_END() // Cpp

