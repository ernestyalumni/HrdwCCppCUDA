#include "Algorithms/HackerRank/InsertionSort.h"
#include "Tools/CaptureCout.h"

#include <boost/test/unit_test.hpp>
#include <string>
#include <vector>

using Algorithms::HackerRank::Sorting::InsertionSort::insertion_sort;
using Algorithms::HackerRank::Sorting::InsertionSort::insertion_sort_1;
using Algorithms::HackerRank::Sorting::InsertionSort::running_time;
using Tools::CaptureCoutFixture;
using std::string;
using std::vector;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(HackerRank)
BOOST_AUTO_TEST_SUITE(Sorting)
BOOST_AUTO_TEST_SUITE(InsertionSort_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(InsertionSort1PrintsSwapsAndInsertions,
	CaptureCoutFixture)
{
	vector<int> arr {1, 2, 4, 5, 3};
	BOOST_TEST_REQUIRE(arr[arr.size() - 1] == 3);
	BOOST_TEST_REQUIRE(arr[arr.size() - 2] == 5);

	insertion_sort_1(arr.size(), arr);

	restore_cout();

  BOOST_TEST(local_oss_.str() == "1 2 4 5 5 \n1 2 4 4 5 \n1 2 3 4 5 ");
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(InsertionSort1PrintsSwapsAndInsertionsExample2,
	CaptureCoutFixture)
{
	vector<int> arr {2, 4, 6, 8, 3};
	BOOST_TEST_REQUIRE(arr[arr.size() - 1] == 3);
	BOOST_TEST_REQUIRE(arr[arr.size() - 2] == 8);

	insertion_sort_1(arr.size(), arr);

	restore_cout();

  BOOST_TEST(local_oss_.str() ==
  	"2 4 6 8 8 \n2 4 6 6 8 \n2 4 4 6 8 \n2 3 4 6 8 ");
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(InsertionSort1PrintsSwapsAndInsertionsTestCase2,
	CaptureCoutFixture)
{
	vector<int> arr {2, 3, 4, 5, 6, 7, 8, 9, 10, 1};
	BOOST_TEST_REQUIRE(arr[arr.size() - 1] == 1);
	BOOST_TEST_REQUIRE(arr[arr.size() - 2] == 10);

	insertion_sort_1(arr.size(), arr);

	restore_cout();

	std::cout << local_oss_.str();

	string expected {"2 3 4 5 6 7 8 9 10 10 \n"};
	expected += "2 3 4 5 6 7 8 9 9 10 \n";
	expected += "2 3 4 5 6 7 8 8 9 10 \n";
	expected += "2 3 4 5 6 7 7 8 9 10 \n";
	expected += "2 3 4 5 6 6 7 8 9 10 \n";
	expected += "2 3 4 5 5 6 7 8 9 10 \n";
	expected += "2 3 4 4 5 6 7 8 9 10 \n";
	expected += "2 3 3 4 5 6 7 8 9 10 \n";
	expected += "2 2 3 4 5 6 7 8 9 10 \n";
	expected += "1 2 3 4 5 6 7 8 9 10 ";

  BOOST_TEST(local_oss_.str() == expected);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(InsertionSortSorts)
{
	int arr[] {4, 1, 3, 5, 6, 2};

	insertion_sort(6, arr);

	BOOST_TEST(arr[0] == 1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(RunningTimeCountsShifts)
{
	{
		vector<int> arr {2, 1, 3, 1, 2};

		BOOST_TEST(running_time(arr) == 4);

		BOOST_TEST(arr[0] == 1);
		BOOST_TEST(arr[1] == 1);
		BOOST_TEST(arr[2] == 2);
		BOOST_TEST(arr[3] == 2);
		BOOST_TEST(arr[4] == 3);
	}
}

// https://www.hackerrank.com/challenges/runningtime/problem?h_r=next-challenge&h_v=zen
// SampleTestCase1
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(RunningTimeCountsNoShiftsOnSortedList)
{
	vector<int> arr {1, 1, 2, 2, 3, 3, 5, 5, 7, 7, 9, 9};

	BOOST_TEST(running_time(arr) == 0);
}

BOOST_AUTO_TEST_SUITE_END() // InsertionSort_tests
BOOST_AUTO_TEST_SUITE_END() // Sorting
BOOST_AUTO_TEST_SUITE_END() // HackerRank
BOOST_AUTO_TEST_SUITE_END() // Algorithms