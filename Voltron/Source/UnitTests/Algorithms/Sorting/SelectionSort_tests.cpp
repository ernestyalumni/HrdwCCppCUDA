#include "Algorithms/Sorting/SelectionSort.h"

#include <boost/test/unit_test.hpp>
#include <vector>

using Algorithms::Sorting::SelectionSort::SelectionSortIterative;
using Algorithms::Sorting::SelectionSort::minimum_value_index;
using Algorithms::Sorting::SelectionSort::selection_sort_recursive;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(SelectionSort_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SelectionSortIterativeSortsCorrectly)
{ 
  std::vector<int> a {12, 11, 13, 5, 6};

  SelectionSortIterative::selection_sort(a);

  BOOST_TEST(a[0] == 5);
  BOOST_TEST(a[1] == 6);
  BOOST_TEST(a[2] == 11);
  BOOST_TEST(a[3] == 12);
  BOOST_TEST(a[4] == 13);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SelectionSortRecursivelySortsCorrectly)
{ 
  std::vector<int> a {12, 11, 13, 5, 6};

  selection_sort_recursive<std::vector<int>, int>(
    a,
    static_cast<std::size_t>(0),
    a.size());

  BOOST_TEST(a[0] == 5);
  BOOST_TEST(a[1] == 6);
  BOOST_TEST(a[2] == 11);
  BOOST_TEST(a[3] == 12);
  BOOST_TEST(a[4] == 13);
}

BOOST_AUTO_TEST_SUITE_END() // SelectionSort_tests
BOOST_AUTO_TEST_SUITE_END() // Algorithms