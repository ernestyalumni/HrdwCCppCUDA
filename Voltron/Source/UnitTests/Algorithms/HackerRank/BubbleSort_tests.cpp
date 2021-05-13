#include "Algorithms/HackerRank/BubbleSort.h"

#include <boost/test/unit_test.hpp>
#include <vector>

using Algorithms::HackerRank::Sorting::BubbleSort::count_swaps;
using std::vector;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(HackerRank)
BOOST_AUTO_TEST_SUITE(Sorting)
BOOST_AUTO_TEST_SUITE(BubbleSort_tests)

/// \ref https://www.hackerrank.com/challenges/ctci-bubble-sort/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=sorting

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BubbleSortSorts)
{
	{
		vector<int> a {6, 4, 1};
		BOOST_TEST_REQUIRE(a[0] == 6);
		BOOST_TEST_REQUIRE(a[2] == 1);

		int result {count_swaps(a)};

		BOOST_TEST(a[0] == 1);
		BOOST_TEST(a[2] == 6);		
	}
	// HackerRank, BubbleSort, Test case 1.
	{
		vector<int> a {3, 2, 1};
		BOOST_TEST_REQUIRE(a[0] == 3);
		BOOST_TEST_REQUIRE(a[2] == 1);

		int result {count_swaps(a)};

		BOOST_TEST(a[0] == 1);
		BOOST_TEST(a[2] == 3);
	}
	// HackerRank, BubbleSort, Test case 3.
	{
		vector<int> a {4, 2, 3, 1};
		BOOST_TEST_REQUIRE(a[0] == 4);
		BOOST_TEST_REQUIRE(a[3] == 1);

		int result {count_swaps(a)};

		BOOST_TEST(a[0] == 1);
		BOOST_TEST(a[3] == 4);
	}
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SortedVectorsStaySorted)
{
	// HackerRank, BubbleSort, Test case 0.
	vector<int> a {1, 2, 3};

	int result {count_swaps(a)};

	BOOST_TEST(a[0] == 1);
	BOOST_TEST(a[1] == 2);
	BOOST_TEST(a[2] == 3);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CountSwapsCountsSwaps)
{
	{
		vector<int> a {6, 4, 1};
		BOOST_TEST_REQUIRE(a[0] == 6);
		BOOST_TEST_REQUIRE(a[2] == 1);

		int result {count_swaps(a)};

		BOOST_TEST(result == 3);
	}
	{
		vector<int> a {3, 2, 1};
		BOOST_TEST_REQUIRE(a[0] == 3);
		BOOST_TEST_REQUIRE(a[2] == 1);

		int result {count_swaps(a)};

		BOOST_TEST(result == 3);
	}
	{
		vector<int> a {4, 2, 3, 1};
		BOOST_TEST_REQUIRE(a[0] == 4);
		BOOST_TEST_REQUIRE(a[3] == 1);

		int result {count_swaps(a)};

		BOOST_TEST(result == 5);
	}
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SortedVectorsResultInZeroSwaps)
{
	vector<int> a {1, 2, 3};

	int result {count_swaps(a)};

	BOOST_TEST(result == 0);
}

BOOST_AUTO_TEST_SUITE_END() // BubbleSort_tests
BOOST_AUTO_TEST_SUITE_END() // Sorting
BOOST_AUTO_TEST_SUITE_END() // HackerRank
BOOST_AUTO_TEST_SUITE_END() // Algorithms