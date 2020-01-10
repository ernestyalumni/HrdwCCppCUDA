//------------------------------------------------------------------------------
// \file Sorting_tests.cpp
//------------------------------------------------------------------------------
#include "Algorithms/BubbleSort.h"
#include "Algorithms/MergeSort.h"
#include "Algorithms/QuickSort.h"
#include "Cpp/Std/TypeTraitsProperties.h"

#include <array>
#include <boost/test/unit_test.hpp>
#include <deque>
#include <forward_list>
#include <list>
#include <sstream>
#include <string>
#include <vector>

using Algorithms::Sorting::Details::naive_single_pass;
using Algorithms::Sorting::Details::single_pass;
using Algorithms::Sorting::Details::single_swap;
using Algorithms::Sorting::QuickSort::Details::partition_from_first;
using Algorithms::Sorting::bubble_sort;
using Algorithms::Sorting::merge_sort;
using Algorithms::Sorting::naive_bubble_sort;
using Algorithms::Sorting::QuickSort::Details::quick_sort_from_first;
using Algorithms::Sorting::QuickSort::quick_sort_from_first;
using Algorithms::Sorting::QuickSort::quick_sort_from_last;
using Std::CompositeTypeTraits;
using Std::PrimaryTypeTraits;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(Sorting_tests)

BOOST_AUTO_TEST_SUITE(BubbleSort_tests)

BOOST_AUTO_TEST_SUITE(Details_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateSingleSwapping)
{
  {
    std::array<unsigned int, 5> a {5, 1, 4, 2, 8};

    single_swap(a, 2, 4);

    BOOST_TEST(a[2] == 8);
    BOOST_TEST(a[4] == 4);
  }
  {
    std::vector<unsigned int> a {5, 1, 4, 2, 8};

    single_swap(a, 2, 4);

    BOOST_TEST(a[2] == 8);
    BOOST_TEST(a[4] == 4);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateContainerProperties)
{
  {
    std::cout << "\n std::array \n";
    PrimaryTypeTraits<std::array<int, 42>> array_primary_type_traits;
    std::cout << array_primary_type_traits << '\n';
    CompositeTypeTraits<std::array<int, 42>> array_composite_type_traits;
    std::cout << array_composite_type_traits << '\n';

    std::cout << "\n std::list \n";
    PrimaryTypeTraits<std::list<int>> list_primary_type_traits;
    std::cout << list_primary_type_traits << '\n';
    CompositeTypeTraits<std::list<int>> list_composite_type_traits;
    std::cout << list_composite_type_traits << '\n';

    std::cout << "\n std::vector \n";
    PrimaryTypeTraits<std::vector<int>> vector_primary_type_traits;
    std::cout << vector_primary_type_traits << '\n';
    CompositeTypeTraits<std::vector<int>> vector_composite_type_traits;
    std::cout << vector_composite_type_traits << '\n';
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateNaiveSinglePass)
{
  {
    std::array<unsigned int, 5> a {5, 1, 4, 2, 8};

    BOOST_TEST(naive_single_pass(a));

    BOOST_TEST(a[0] == 1);
    BOOST_TEST(a[1] == 4);
    BOOST_TEST(a[2] == 2);
    BOOST_TEST(a[3] == 5);
    BOOST_TEST(a[4] == 8);
  }
  {
    std::vector<unsigned int> a {5, 1, 4, 2, 8};

    BOOST_TEST(naive_single_pass(a));

    BOOST_TEST(a[2] == 2);
    BOOST_TEST(a[4] == 8);

    BOOST_TEST(naive_single_pass(a));

    BOOST_TEST(a[0] == 1);
    BOOST_TEST(a[1] == 2);
    BOOST_TEST(a[2] == 4);
    BOOST_TEST(a[3] == 5);
    BOOST_TEST(a[4] == 8);

    BOOST_TEST(!naive_single_pass(a));
  }
  {
    std::vector<unsigned int> a {21, 4, 1, 3, 9, 20, 25, 6, 21, 14};

    BOOST_TEST(naive_single_pass(a));

    BOOST_TEST(a[2] == 3);
    BOOST_TEST(a[4] == 20);

    BOOST_TEST(naive_single_pass(a));

    BOOST_TEST(a[0] == 1);
    BOOST_TEST(a[1] == 3);
    BOOST_TEST(a[2] == 4);
    BOOST_TEST(a[3] == 9);
    BOOST_TEST(a[4] == 20);

    BOOST_TEST(naive_single_pass(a));

    BOOST_TEST(a[0] == 1);
    BOOST_TEST(a[1] == 3);
    BOOST_TEST(a[2] == 4);
    BOOST_TEST(a[3] == 9);
    BOOST_TEST(a[4] == 6);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateSinglePass)
{
  {
    std::array<unsigned int, 5> a {5, 1, 4, 2, 8};

    single_pass(a, 2);

    BOOST_TEST(a[0] == 1);
    BOOST_TEST(a[1] == 4);
    BOOST_TEST(a[2] == 2);
    BOOST_TEST(a[3] == 5);
    BOOST_TEST(a[4] == 8);
  }
  {
    std::vector<unsigned int> a {5, 1, 4, 2, 8};

    single_pass(a, 3);

    BOOST_TEST(a[2] == 5);
    BOOST_TEST(a[4] == 8);

    single_pass(a, 4);

    BOOST_TEST(a[0] == 1);
    BOOST_TEST(a[1] == 4);
    BOOST_TEST(a[2] == 5);
    BOOST_TEST(a[3] == 2);
    BOOST_TEST(a[4] == 8);
  }
}

BOOST_AUTO_TEST_SUITE_END() // Details_tests

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateNaiveBubbleSort)
{
  {
    std::array<unsigned int, 5> a {5, 1, 4, 2, 8};

    naive_bubble_sort(a);

    BOOST_TEST(a[0] == 1);
    BOOST_TEST(a[1] == 2);
    BOOST_TEST(a[2] == 4);
    BOOST_TEST(a[3] == 5);
    BOOST_TEST(a[4] == 8);
  }
  {
    std::vector<unsigned int> a {5, 1, 4, 2, 8};

    naive_bubble_sort(a);

    BOOST_TEST(a[2] == 4);
    BOOST_TEST(a[4] == 8);

    naive_bubble_sort(a);

    BOOST_TEST(a[0] == 1);
    BOOST_TEST(a[1] == 2);
    BOOST_TEST(a[2] == 4);
    BOOST_TEST(a[3] == 5);
    BOOST_TEST(a[4] == 8);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateBubbleSort)
{
  {
    std::array<unsigned int, 5> a {5, 1, 4, 2, 8};

    bubble_sort(a);

    BOOST_TEST(a[0] == 1);
    BOOST_TEST(a[1] == 2);
    BOOST_TEST(a[2] == 4);
    BOOST_TEST(a[3] == 5);
    BOOST_TEST(a[4] == 8);
  }
  {
    std::vector<unsigned int> a {5, 1, 4, 2, 8};

    bubble_sort(a);

    BOOST_TEST(a[2] == 4);
    BOOST_TEST(a[4] == 8);

    bubble_sort(a);

    BOOST_TEST(a[0] == 1);
    BOOST_TEST(a[1] == 2);
    BOOST_TEST(a[2] == 4);
    BOOST_TEST(a[3] == 5);
    BOOST_TEST(a[4] == 8);
  }
}

BOOST_AUTO_TEST_SUITE_END() // BubbleSort_tests

BOOST_AUTO_TEST_SUITE(MergeSort_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateMergeSort)
{
  {
    std::array<unsigned int, 6> a {12, 11, 13, 5, 6, 7};

    merge_sort(a);

    BOOST_TEST(a[0] == 5);
    BOOST_TEST(a[1] == 6);
    BOOST_TEST(a[2] == 7);
    BOOST_TEST(a[3] == 11);
    BOOST_TEST(a[4] == 12);
    BOOST_TEST(a[5] == 13);
  }
  {
    std::vector<unsigned int> a {12, 11, 13, 5, 6, 7};

    merge_sort(a);

    BOOST_TEST(a[0] == 5);
    BOOST_TEST(a[1] == 6);
    BOOST_TEST(a[2] == 7);
    BOOST_TEST(a[3] == 11);
    BOOST_TEST(a[4] == 12);
    BOOST_TEST(a[5] == 13);
  }
}

BOOST_AUTO_TEST_SUITE_END() // MergeSort_tests

BOOST_AUTO_TEST_SUITE(QuickSort_tests)

BOOST_AUTO_TEST_SUITE(Details_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateQuickSortPartitionFromLast)
{
  {
    std::vector<int> a {8, 3, 1, 7, 0, 10, 2};

    std::size_t new_pivot {partition_from_first(a, 0, a.size() - 1)};

    BOOST_TEST(new_pivot == 4);

    BOOST_TEST(a[0] == 0);
    BOOST_TEST(a[1] == 3);
    BOOST_TEST(a[2] == 1);
    BOOST_TEST(a[3] == 7);
    BOOST_TEST(a[4] == 8);
    BOOST_TEST(a[5] == 10);
    BOOST_TEST(a[6] == 2);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateQuickSortFromLast)
{
  {
    std::vector<int> a {8, 3, 1, 7, 0, 10, 2};

    quick_sort_from_first(a, 0, a.size());

    BOOST_TEST(a[0] == 0);
    BOOST_TEST(a[1] == 1);
    BOOST_TEST(a[2] == 2);
    BOOST_TEST(a[3] == 3);
    BOOST_TEST(a[4] == 7);
    BOOST_TEST(a[5] == 8);
    BOOST_TEST(a[6] == 10);   
  }
}


BOOST_AUTO_TEST_SUITE_END() // Details_tests

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateQuickSortFromFirst)
{
  {
    std::vector<int> a {12, 11, 13, 5, 6, 7};
    
    quick_sort_from_first(a);
    
    BOOST_TEST(a[0] == 5);
    BOOST_TEST(a[1] == 6);
    BOOST_TEST(a[2] == 7);
    BOOST_TEST(a[3] == 11);
    BOOST_TEST(a[4] == 12);
    BOOST_TEST(a[5] == 13);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateQuickSortFromLast)
{
    std::array<unsigned int, 7> a {8, 3, 1, 7, 0, 10, 2};

    quick_sort_from_last(a);

    BOOST_TEST(a[0] == 0);
    BOOST_TEST(a[1] == 1);
    BOOST_TEST(a[2] == 2);
    BOOST_TEST(a[3] == 3);
    BOOST_TEST(a[4] == 7);
    BOOST_TEST(a[5] == 8);
    BOOST_TEST(a[6] == 10);

  {
    std::vector<int> a {12, 11, 13, 5, 6, 7};
    
    quick_sort_from_last(a);
    
    BOOST_TEST(a[0] == 5);
    BOOST_TEST(a[1] == 6);
    BOOST_TEST(a[2] == 7);
    BOOST_TEST(a[3] == 11);
    BOOST_TEST(a[4] == 12);
    BOOST_TEST(a[5] == 13);
  }
}


BOOST_AUTO_TEST_SUITE_END() // QuickSort_tests

BOOST_AUTO_TEST_SUITE_END() // Sorting_tests
BOOST_AUTO_TEST_SUITE_END() // Algorithms