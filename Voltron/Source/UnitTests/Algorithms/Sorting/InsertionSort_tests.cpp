#include "Algorithms/Sorting/InsertionSort.h"

#include <boost/test/unit_test.hpp>

using Algorithms::Sorting::InsertionSort::insertion_sort;
using Algorithms::Sorting::InsertionSort::insertion_sort_optimized;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(Sorting)
BOOST_AUTO_TEST_SUITE(InsertionSort_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(InsertionSortSortsCStyleIntArrays)
{
  int arr[] {12, 11, 13, 5, 6};
  const int n {sizeof(arr) / sizeof(arr[0])};  

  insertion_sort(arr, n);
  BOOST_TEST(arr[0] == 5);
  BOOST_TEST(arr[1] == 6);
  BOOST_TEST(arr[2] == 11);
  BOOST_TEST(arr[3] == 12);
  BOOST_TEST(arr[4] == 13);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(InsertionSortOptimizedSortsCStyleIntArrays)
{
  int arr[] {12, 11, 13, 5, 6};
  const int n {sizeof(arr) / sizeof(arr[0])};  

  insertion_sort_optimized(arr, n);
  BOOST_TEST(arr[0] == 5);
  BOOST_TEST(arr[1] == 6);
  BOOST_TEST(arr[2] == 11);
  BOOST_TEST(arr[3] == 12);
  BOOST_TEST(arr[4] == 13);
}

BOOST_AUTO_TEST_SUITE_END() // InsertionSort_tests

BOOST_AUTO_TEST_SUITE_END() // Sorting
BOOST_AUTO_TEST_SUITE_END() // Algorithms