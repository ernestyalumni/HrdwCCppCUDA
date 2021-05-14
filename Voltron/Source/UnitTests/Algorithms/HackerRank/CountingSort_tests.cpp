/// \ref https://www.hackerrank.com/challenges/countingsort1/problem

#include "Algorithms/HackerRank/CountingSort.h"

#include <boost/test/unit_test.hpp>
#include <vector>

using Algorithms::HackerRank::Sorting::CountingSort::counting_sort_frequency;
using std::vector;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(HackerRank)
BOOST_AUTO_TEST_SUITE(Sorting)
BOOST_AUTO_TEST_SUITE(CountingSort_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CountingSortFrequencyCountsFrequencies)
{
	vector<int> arr {
		63, 25, 73, 1, 98, 73, 56, 84, 86, 57, 16, 83, 8, 25, 81, 56, 9, 53, 98, 67,
		99, 12, 83, 89, 80, 91, 39, 86, 76, 85, 74, 39, 25, 90, 59, 10, 94, 32, 44,
		3, 89, 30, 27, 79, 46, 96, 27, 32, 18, 21, 92, 69, 81, 40, 40, 34, 68, 78,
		24, 87, 42, 69, 23, 41, 78, 22, 6, 90, 99, 89, 50, 30, 20, 1, 43, 3, 70, 95,
		33, 46, 44, 9, 69, 48, 33, 60, 65, 16, 82, 67, 61, 32, 21, 79, 75, 75, 13,
		87, 70, 33};

	const vector<int> frequency_result {counting_sort_frequency(arr)};

	BOOST_TEST(frequency_result[0] == 0);
	BOOST_TEST(frequency_result[1] == 2);
	BOOST_TEST(frequency_result[2] == 0);
	BOOST_TEST(frequency_result[3] == 2);
	BOOST_TEST(frequency_result[4] == 0);
	BOOST_TEST(frequency_result[5] == 0);
	BOOST_TEST(frequency_result[6] == 1);
}

BOOST_AUTO_TEST_SUITE_END() // CountingSort_tests
BOOST_AUTO_TEST_SUITE_END() // Sorting
BOOST_AUTO_TEST_SUITE_END() // HackerRank
BOOST_AUTO_TEST_SUITE_END() // Algorithms