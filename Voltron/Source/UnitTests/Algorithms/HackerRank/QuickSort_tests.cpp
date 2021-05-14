/// \ref https://www.hackerrank.com/challenges/quicksort1/problem?h_r=next-challenge&h_v=zen

#include "Algorithms/HackerRank/QuickSort.h"

#include <boost/test/unit_test.hpp>
#include <vector>

#include <iostream>

using Algorithms::HackerRank::Sorting::QuickSort::quick_sort_partition;
using std::vector;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(HackerRank)
BOOST_AUTO_TEST_SUITE(Sorting)
BOOST_AUTO_TEST_SUITE(QuickSort_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(QuickSortPartitionPartitions)
{
	vector<int> arr {4, 5, 3, 7, 2};

	const vector<int> partition_result {quick_sort_partition(arr)};

	BOOST_TEST(partition_result[2] == 4);
	BOOST_TEST((partition_result[0] == 2 && partition_result[1] == 3 ||
		partition_result[0] == 3 && partition_result[1] == 2));
	BOOST_TEST((partition_result[3] == 5 && partition_result[4] == 7 ||
		partition_result[3] == 7 && partition_result[4] == 5));
}

BOOST_AUTO_TEST_SUITE_END() // QuickSort_tests
BOOST_AUTO_TEST_SUITE_END() // Sorting
BOOST_AUTO_TEST_SUITE_END() // HackerRank
BOOST_AUTO_TEST_SUITE_END() // Algorithms