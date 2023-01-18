#include "DataStructures/Arrays/DynamicArray.h"

#include <boost/test/unit_test.hpp>
#include <cstddef> // std::size_t
#include <string>
#include <utility> // std::move

using DataStructures::Arrays::DynamicArray;
using DataStructures::Arrays::PrimitiveDynamicArray;
using std::string;

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

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithStringType)
{
  DynamicArray<string> a {};

  a.append("ab");
  a.append("bc");
  a.append("cd");
  a.append("de");
  a.append("ef");
  a.append("fg");
  a.append("gh");
  a.append("hi");

  BOOST_TEST(a[0] == "ab");
  BOOST_TEST(a[1] == "bc");
  BOOST_TEST(a[7] == "hi");
}

BOOST_AUTO_TEST_SUITE_END() // DynamicArray_tests

BOOST_AUTO_TEST_SUITE(PrimitiveDynamicArray_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CopyConstructorCopiesValuesAndLeavesSourceUnchanged)
{
  PrimitiveDynamicArray<int> a (3);
  a.append(35);
  a.append(75);
  PrimitiveDynamicArray<int> b {a};

  BOOST_TEST(b[0] == 35);
  BOOST_TEST(b[1] == 75);

  BOOST_TEST(a[0] == 35);
  BOOST_TEST(a[1] == 75);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CopyAssignmentCopiesValuesAndLeavesSourceUnchanged)
{
  PrimitiveDynamicArray<int> a (3);
  a.append(35);
  a.append(75);
  PrimitiveDynamicArray<int> b (42);
  b.append(69);
  BOOST_TEST(b[0] == 69);

  b = a;

  BOOST_TEST(b[0] == 35);
  BOOST_TEST(b[1] == 75);

  BOOST_TEST(a[0] == 35);
  BOOST_TEST(a[1] == 75);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MoveConstructorMovesValuesAndEmptiesSource)
{
  PrimitiveDynamicArray<int> a (3);
  a.append(35);
  a.append(75);
  PrimitiveDynamicArray<int> b {std::move(a)};

  BOOST_TEST(b[0] == 35);
  BOOST_TEST(b[1] == 75);

  BOOST_TEST(!a.has_data());
  BOOST_TEST(a.size() == 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MoveAssignmentMovesValuesAndEmptiesSource)
{
  PrimitiveDynamicArray<int> a (3);
  a.append(35);
  a.append(75);

  PrimitiveDynamicArray<int> b {};
  b.append(420);

  b = std::move(a);

  BOOST_TEST(b[0] == 35);
  BOOST_TEST(b[1] == 75);

  BOOST_TEST(!a.has_data());
  BOOST_TEST(a.size() == 0);
}

BOOST_AUTO_TEST_SUITE_END() // PrimitiveDynamicArray_tests

BOOST_AUTO_TEST_SUITE_END() // Arrays
BOOST_AUTO_TEST_SUITE_END() // DataStructures