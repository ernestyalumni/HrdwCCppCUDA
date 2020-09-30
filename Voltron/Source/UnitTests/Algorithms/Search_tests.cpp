//------------------------------------------------------------------------------
// \file Search_tests.cpp
//------------------------------------------------------------------------------
#include "Algorithms/BinarySearch.h"
#include "Algorithms/BubbleSort.h"
#include "Algorithms/MergeSort.h"
#include "Algorithms/QuickSort.h"

#include <array>
#include <boost/test/unit_test.hpp>
#include <deque>
#include <forward_list>
#include <list>
#include <string>
#include <vector>

using Algorithms::Search::Details::calculate_midpoint;
using Algorithms::Search::Details::compare_partition;
using Algorithms::Search::Details::binary_search_iteration;
using Algorithms::Search::binary_search;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(Search_tests)
BOOST_AUTO_TEST_SUITE(Binary_Search_tests)

BOOST_AUTO_TEST_SUITE(Details_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateCalculateMidpoint)
{
  std::size_t r {5};
  std::size_t l {3};
  
  // Don't design code that has to static cast a size_t into an int.
  //const long long L {r - l + 1};

  //BOOST_TEST(L == -1);
  {
    const auto result = calculate_midpoint(r, l);
    BOOST_TEST(!result.has_value());
  }
  {
    const auto result = calculate_midpoint(l, r);
    BOOST_TEST(result.has_value());
    BOOST_TEST(result.value() == 4);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateComparePartition)
{
  {
    const auto result = compare_partition(11, 11, 3, 0, 6);
    BOOST_TEST(static_cast<bool>(result));
    BOOST_TEST(result.value().first);
    BOOST_TEST(result.value().second.first == 3);
    BOOST_TEST(result.value().second.second == 3);
  }
  {
    const auto result = compare_partition(11, 10, 3, 0, 6);
    BOOST_TEST(static_cast<bool>(result));
    BOOST_TEST(!result.value().first);
    BOOST_TEST(result.value().second.first == 0);
    BOOST_TEST(result.value().second.second == 2);
  }
  {
    const auto result = compare_partition(11, 12, 3, 0, 6);
    BOOST_TEST(static_cast<bool>(result));
    BOOST_TEST(!result.value().first);
    BOOST_TEST(result.value().second.first == 4);
    BOOST_TEST(result.value().second.second == 6);
  }
  {
    const auto result = compare_partition(1, 0, 0, 0, 1);
    BOOST_TEST(!static_cast<bool>(result));
  }
  {
    const auto result = compare_partition(1, 2, 0, 0, 1);
    BOOST_TEST(result.has_value());
    BOOST_TEST(!result.value().first);
    BOOST_TEST(result.value().second.first == 1);
    BOOST_TEST(result.value().second.second == 1);
  }
  {
    const auto result = compare_partition(29, 30, 6, 6, 6);
    BOOST_TEST(!result.has_value());
  }
}

BOOST_AUTO_TEST_SUITE_END() // Details_tests

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateBinarySearch)
{ 
  std::vector<int> sorted_vector {1, 3, 9, 11, 15, 19, 29};
  {
    const auto result = binary_search(sorted_vector, 25);
    BOOST_TEST(!result.has_value());
  }
  {
    const auto result = binary_search(sorted_vector, -1);
    BOOST_TEST(!result.has_value());
  }
  {
    const auto result = binary_search(sorted_vector, 1);
    BOOST_TEST(result.has_value());
    BOOST_TEST(result.value() == 0);
  }
  {
    const auto result = binary_search(sorted_vector, 3);
    BOOST_TEST(result.has_value());
    BOOST_TEST(result.value() == 1);
  }
  {
    const auto result = binary_search(sorted_vector, 19);
    BOOST_TEST(result.has_value());
    BOOST_TEST(result.value() == 5);
  }
  {
    const auto result = binary_search(sorted_vector, 29);
    BOOST_TEST(result.has_value());
    BOOST_TEST(result.value() == 6);
  }
}

BOOST_AUTO_TEST_SUITE_END() // Binary_Search_tests
BOOST_AUTO_TEST_SUITE_END() // Search_tests
BOOST_AUTO_TEST_SUITE_END() // Algorithms