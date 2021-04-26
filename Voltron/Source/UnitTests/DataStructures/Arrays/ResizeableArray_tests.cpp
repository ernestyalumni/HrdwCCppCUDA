#include "DataStructures/Arrays/ResizeableArray.h"

#include <boost/test/unit_test.hpp>
#include <utility>

using DataStructures::Arrays::ResizeableArray;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Arrays)
BOOST_AUTO_TEST_SUITE(ResizeableArray_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
  {
    ResizeableArray<int> array;
    BOOST_TEST(array.has_data());
    BOOST_TEST(array.size() == 0);
    BOOST_TEST(array.capacity() == array.default_size_);
  }
  {
    ResizeableArray<int> array {};
    BOOST_TEST(array.has_data());
    BOOST_TEST(array.size() == 0);
    BOOST_TEST(array.capacity() == array.default_size_);  
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AppendsPastDefaultSize)
{
  ResizeableArray<int> array;
  BOOST_TEST(array.has_data());

  for (int i {0}; i < array.default_size_; ++i)
  {
    array.append(i * 2);
    BOOST_TEST(array.capacity() == array.default_size_);
    BOOST_TEST(array.size() == i + 1);
    BOOST_TEST(array[i] == i * 2);
  }

  int i {array.default_size_};
  array.append(i * 2);
  BOOST_TEST(array.capacity() == array.default_size_ * 2);
  BOOST_TEST(array.size() == i + 1);
  BOOST_TEST(array[i] == i * 2);

  ++i;
  array.append(i * 2);
  BOOST_TEST(array.capacity() == array.default_size_ * 2);
  BOOST_TEST(array.size() == i + 1);
  BOOST_TEST(array[i] == i * 2);

  BOOST_TEST(array.has_data());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsFromInitializerList)
{
  ResizeableArray<int> array {16, 14, 10, 8, 7, 9, 3, 2, 4, 1};
  BOOST_TEST(array.has_data());

  BOOST_TEST(array[0] == 16);
  BOOST_TEST(array[1] == 14);  

  BOOST_TEST(array.size() == 10);
  BOOST_TEST(array.capacity() == 10);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CopyConstructsForExactDataCopy)
{
  ResizeableArray<int> array {42, 43, 44};
  BOOST_TEST(array.has_data());
  BOOST_TEST(array.size() == 3);
  BOOST_TEST(array.capacity() == 3);

  const ResizeableArray<int> array_copy {array};
  BOOST_TEST(array_copy.has_data());

  BOOST_TEST(array_copy.size() == 3);
  BOOST_TEST(array_copy.capacity() == 3);

  for (int i {0}; i < 3; ++i)
  {
    BOOST_TEST(array[i] == 42 + i);
    BOOST_TEST(array_copy[i] == 42 + i);
  }

  BOOST_TEST(true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MoveConstructorMovesDataAndSetsOriginal)
{
  ResizeableArray<int> array {42, 43, 44};
  BOOST_TEST(array.has_data());

  {
    ResizeableArray<int> moved_array {std::move(array)};

    BOOST_TEST(array.size() == 0);
    BOOST_TEST(array.capacity() == 3);
    BOOST_TEST(!array.has_data());

    BOOST_TEST(moved_array.size() == 3);
    BOOST_TEST(moved_array.capacity() == 3);
    BOOST_TEST(moved_array.has_data());

    for (int i {0}; i < 3; ++i)
    {
      BOOST_TEST(moved_array[i] == 42 + i);
    }
  }  

  BOOST_TEST(!array.has_data());
  BOOST_TEST(array.size() == 0);
  BOOST_TEST(array.capacity() == 3);
}

BOOST_AUTO_TEST_SUITE_END() // ResizeableArray_tests
BOOST_AUTO_TEST_SUITE_END() // Arrays
BOOST_AUTO_TEST_SUITE_END() // DataStructures