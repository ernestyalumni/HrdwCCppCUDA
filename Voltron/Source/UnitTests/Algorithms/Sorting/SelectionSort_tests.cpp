#include "Algorithms/Sorting/SelectionSort.h"

#include <boost/test/unit_test.hpp>
#include <vector>

using Algorithms::Sorting::SelectionSort;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(SelectionSort_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SelectionSortSortsCorrectly)
{ 
  std::vector<int> a {12, 11, 13, 5, 6};

  SelectionSort::selection_sort(a);

  BOOST_TEST(a[0] == 5);
  BOOST_TEST(a[1] == 6);
  BOOST_TEST(a[2] == 11);
  BOOST_TEST(a[3] == 12);
  BOOST_TEST(a[4] == 13);
}

BOOST_AUTO_TEST_SUITE_END() // SelectionSort_tests
BOOST_AUTO_TEST_SUITE_END() // Algorithms