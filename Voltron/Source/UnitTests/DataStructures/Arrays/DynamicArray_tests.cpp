#include "DataStructures/Arrays/DynamicArray.h"

#include <boost/test/unit_test.hpp>
#include <cstddef> // std::size_t

using DataStructures::Arrays::DynamicArray;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Arrays)
BOOST_AUTO_TEST_SUITE(DynamicArray_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
  DynamicArray<double> a {};
  BOOST_TEST(a.has_data());
  BOOST_TEST(a.size() == 0);
  BOOST_TEST(a.capacity() == a.default_capacity_);  
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithSizeOnly)
{
  {
    DynamicArray<int> a {0};
    BOOST_TEST(a.has_data());
    BOOST_TEST(a.size() == 0);
    BOOST_TEST(a.capacity() == a.default_capacity_);
  }
  {
    DynamicArray<int> a {1};
    BOOST_TEST(a.has_data());
    BOOST_TEST(a.size() == 1);
    BOOST_TEST(a.capacity() == a.default_capacity_);
  }
  {
    DynamicArray<int> a {9};
    BOOST_TEST(a.has_data());
    BOOST_TEST(a.size() == 9);
    BOOST_TEST(a.capacity() == 9);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AppendsPastDefaultSize)
{
  DynamicArray<int> a {};

  for (std::size_t i {0}; i < a.default_capacity_; ++i)
  {
    a.append(i * 2);
    BOOST_TEST(a.capacity() == a.default_capacity_);
    BOOST_TEST(a.size() == i + 1);
    BOOST_TEST(a[i] == i * 2);
  }

  int i {a.default_capacity_};
  a.append(i * 2);
  BOOST_TEST(a.capacity() == a.default_capacity_ * 2);
  BOOST_TEST(a.size() == i + 1);
  BOOST_TEST(a[i] == i * 2);

  ++i;
  a.append(i * 2);
  BOOST_TEST(a.capacity() == a.default_capacity_ * 2);
  BOOST_TEST(a.size() == i + 1);
  BOOST_TEST(a[i] == i * 2);
}

BOOST_AUTO_TEST_SUITE_END() // DynamicArray_tests
BOOST_AUTO_TEST_SUITE_END() // Arrays
BOOST_AUTO_TEST_SUITE_END() // DataStructures