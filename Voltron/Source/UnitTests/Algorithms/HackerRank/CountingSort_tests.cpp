/// \ref https://www.hackerrank.com/challenges/countingsort1/problem

#include "Algorithms/HackerRank/CountingSort.h"
#include "Tools/CaptureCout.h"

#include <boost/test/unit_test.hpp>
#include <string>
#include <vector>

using Algorithms::HackerRank::Sorting::CountingSort::count_sort;
using Algorithms::HackerRank::Sorting::CountingSort::counting_sort;
using Algorithms::HackerRank::Sorting::CountingSort::counting_sort_frequency;
using Tools::CaptureCoutFixture;
using std::string;
using std::vector;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(HackerRank)
BOOST_AUTO_TEST_SUITE(Sorting)
BOOST_AUTO_TEST_SUITE(CountingSort_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CountingSortFrequencyCountsFrequencies)
{
	{
		vector<int> arr {
			63, 25, 73, 1, 98, 73, 56, 84, 86, 57, 16, 83, 8, 25, 81, 56, 9, 53, 98,
			67, 99, 12, 83, 89, 80, 91, 39, 86, 76, 85, 74, 39, 25, 90, 59, 10, 94,
			32, 44, 3, 89, 30, 27, 79, 46, 96, 27, 32, 18, 21, 92, 69, 81, 40, 40, 34,
			68, 78, 24, 87, 42, 69, 23, 41, 78, 22, 6, 90, 99, 89, 50, 30, 20, 1, 43,
			3, 70, 95, 33, 46, 44, 9, 69, 48, 33, 60, 65, 16, 82, 67, 61, 32, 21, 79,
			75, 75, 13, 87, 70, 33};

		const vector<int> frequency_result {counting_sort_frequency(arr)};

		BOOST_TEST(frequency_result[0] == 0);
		BOOST_TEST(frequency_result[1] == 2);
		BOOST_TEST(frequency_result[2] == 0);
		BOOST_TEST(frequency_result[3] == 2);
		BOOST_TEST(frequency_result[4] == 0);
		BOOST_TEST(frequency_result[5] == 0);
		BOOST_TEST(frequency_result[6] == 1);
	}

	{
		vector<int> arr {
			63, 54, 17, 78, 43, 70, 32, 97, 16, 94, 74, 18, 60, 61, 35, 83, 13, 56,
			75, 52, 70, 12, 24, 37, 17, 0, 16, 64, 34, 81, 82, 24, 69, 2, 30, 61, 83,
			37, 97, 16, 70, 53, 0, 61, 12, 17, 97, 67, 33, 30, 49, 70, 11, 40, 67, 94,
			84, 60, 35, 58, 19, 81, 16, 14, 68, 46, 42, 81, 75, 87, 13, 84, 33, 34,
			14, 96, 7, 59, 17, 98, 79, 47, 71, 75, 8, 27, 73, 66, 64, 12, 29, 35, 80,
			78, 80, 6, 5, 24, 49, 82};

		const vector<int> frequency_result {counting_sort_frequency(arr)};

		BOOST_TEST(frequency_result[0] == 2);
		BOOST_TEST(frequency_result[1] == 0);
		BOOST_TEST(frequency_result[2] == 1);
		BOOST_TEST(frequency_result[3] == 0);
		BOOST_TEST(frequency_result[4] == 0);
		BOOST_TEST(frequency_result[5] == 1);
		BOOST_TEST(frequency_result[6] == 1);
	}
}

// \ref https://www.hackerrank.com/challenges/countingsort2/problem?h_r=next-challenge&h_v=zen
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CountingSortSorts)
{
	// Test case 4
	{
		vector<int> arr {19, 10, 12, 10, 24, 25, 22};

		const vector<int> result {counting_sort(arr)};

		BOOST_TEST(result[0] == 10);
		BOOST_TEST(result[1] == 10);
		BOOST_TEST(result[2] == 12);
		BOOST_TEST(result[3] == 19);
		BOOST_TEST(result[4] == 22);
		BOOST_TEST(result[5] == 24);
		BOOST_TEST(result[6] == 25);
	}
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(CountSortSortsKeyValue, CaptureCoutFixture)
{
	vector<vector<string>> arr {
		{"1", "e"},
		{"2", "a"},
		{"1", "b"},
		{"3", "a"},
		{"4", "f"},
		{"1", "f"},
		{"2", "a"},
		{"1", "e"},
		{"1", "b"},
		{"1", "c"}};

	count_sort(arr);

	restore_cout();

  BOOST_TEST(local_oss_.str() == "- - f e b c - a - - ");
}

BOOST_AUTO_TEST_SUITE_END() // CountingSort_tests
BOOST_AUTO_TEST_SUITE_END() // Sorting
BOOST_AUTO_TEST_SUITE_END() // HackerRank
BOOST_AUTO_TEST_SUITE_END() // Algorithms