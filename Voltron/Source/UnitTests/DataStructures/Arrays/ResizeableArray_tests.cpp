#include "DataStructures/Arrays/ResizeableArray.h"

#include <boost/test/unit_test.hpp>
#include <utility>

using DataStructures::Arrays::ResizeableArray;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Arrays)
BOOST_AUTO_TEST_SUITE(ResizeableArray_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(GetSizeOfCStyleArrays)
{
  constexpr int default_size {8};

  // Request block of memory of size default_size
  // https://stackoverflow.com/questions/56931227/when-do-you-need-to-delete-c-style-arrays/56931247
  int* items {new int[default_size]};
  
  int* int_ptr {nullptr};

  // Wrong, No.
  //int items[default_size];
  //items = (new int[default_size]);

  BOOST_TEST(sizeof(items[0]) == sizeof(int));

  // Standard doesn't require implementation to remember element requested
  // through new.
  // Wrong, No.
  //BOOST_TEST(sizeof(items) / sizeof(items[0]) == default_size);

  items[0] == 69;
  items[1] == 42;

  // Same size as a pointer.
  BOOST_TEST(sizeof(items) == sizeof(int_ptr));
  BOOST_TEST(sizeof(items) == sizeof(int*));
  BOOST_TEST(sizeof(items[0]) == sizeof(int));

  // Release block of memory pointed by pointer-variable.
  // cf. https://www.softwaretestinghelp.com/new-delete-operators-in-cpp/
  // If delete items, items point to first element of array and this statement
  // will only delete first element of array. Using subscript "[]", indicates
  // variable whose memory is being freed is an array and all memory allocated
  // is to be freed.
  delete[] items;
}

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
BOOST_AUTO_TEST_CASE(
  ConstructsWithTotalNumberOfElementsUsingParenthesesAndInitialized)
{
  constexpr size_t N {5};
  ResizeableArray<int> array (N);

  BOOST_TEST(array.capacity() == N);

  // Test that the ctor initializes all elements to 0; this is because the empty
  // initializer list induces this.
  for (size_t i {0}; i < N; ++i)
  {
    BOOST_TEST(array[i] == 0);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithInitializerListOf1Element)
{
  const ResizeableArray<int> array {5};
  BOOST_TEST(array.size() == 1);
  BOOST_TEST(array[0] == 5);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithInitializerListOfMultipleElements)
{
  const ResizeableArray<int> array {5, 2, 8, 4, 9, 1};
  BOOST_TEST(array.size() == 6);
  BOOST_TEST(array[0] == 5);
  BOOST_TEST(array[1] == 2);
  BOOST_TEST(array[2] == 8);
  BOOST_TEST(array[3] == 4);
  BOOST_TEST(array[4] == 9);
  BOOST_TEST(array[5] == 1);
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
BOOST_AUTO_TEST_CASE(ResizingArrayResizes)
{
  ResizeableArray<int> array;
  BOOST_TEST(array.size() == 0);
  BOOST_TEST(array.capacity() == array.default_size_);  
  BOOST_TEST(array.capacity() > array.size());  
  
  array.append(42);
  BOOST_TEST(array[0] == 42);
  BOOST_TEST(array.size() == 1);
  BOOST_TEST(array.capacity() == array.default_size_);  

  array.append(43);
  BOOST_TEST(array[1] == 43);
  BOOST_TEST(array.size() == 2);
  BOOST_TEST(array.capacity() == array.default_size_);  

  for (int i {0}; i < (array.capacity() - 2); ++i)
  {
    array.append(44 + i);
    BOOST_TEST(array[i + 2] == 44 + i);
    BOOST_TEST(array.size() == 3 + i);
    BOOST_TEST(array.capacity() == array.default_size_);
  }

  array.append(50);
  BOOST_TEST(array[8] == 50);
  BOOST_TEST(array.size() == 9);
  BOOST_TEST(array.capacity() == array.default_size_ * 2);
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