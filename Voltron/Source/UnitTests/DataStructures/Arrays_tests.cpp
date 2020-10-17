//------------------------------------------------------------------------------
/// \file Arrays_tests.cpp
/// \date 20201015 15, 17:21 complete
//------------------------------------------------------------------------------
#include "DataStructures/Arrays.h"

#include <boost/test/unit_test.hpp>
#include <vector>

using DataStructures::Arrays::LeetCode::check_if_double_exists;
using DataStructures::Arrays::LeetCode::fastest_find_sorted_arrays_median;
using DataStructures::Arrays::LeetCode::fastest_replace_with_greatest_on_right;
using DataStructures::Arrays::LeetCode::find_sorted_arrays_median;
using DataStructures::Arrays::LeetCode::insertion_sort;
using DataStructures::Arrays::LeetCode::replace_with_greatest_on_right;
using DataStructures::Arrays::LeetCode::valid_mountain_array;
using DataStructures::Arrays::ResizeableArray;
using DataStructures::Arrays::rotate_left;
using std::vector;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Arrays_tests)
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
BOOST_AUTO_TEST_CASE(ResizeableArrayResizes)
{
  ResizeableArray<int> array;
  BOOST_TEST(array.size() == 0);
  BOOST_TEST(array.capacity() == array.default_size_);  
  BOOST_TEST(array.capacity() > array.size());  
  
  array.append(42);
  BOOST_TEST(array.get(0) == 42);
  BOOST_TEST(array.size() == 1);
  BOOST_TEST(array.capacity() == array.default_size_);  

  array.append(43);
  BOOST_TEST(array.get(1) == 43);
  BOOST_TEST(array.size() == 2);
  BOOST_TEST(array.capacity() == array.default_size_);  

  for (int i {0}; i < (array.capacity() - 2); ++i)
  {
    array.append(44 + i);
    BOOST_TEST(array.get(i + 2) == 44 + i);
    BOOST_TEST(array.size() == 3 + i);
    BOOST_TEST(array.capacity() == array.default_size_);
  }

  array.append(50);
  BOOST_TEST(array.get(8) == 50);
  BOOST_TEST(array.size() == 9);
  BOOST_TEST(array.capacity() == array.default_size_ * 2);
}

BOOST_AUTO_TEST_SUITE_END() // ResizeableArray_tests

BOOST_AUTO_TEST_SUITE(RotateLeft)

const vector<int> sample_input {1, 2, 3, 4, 5};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(RotateLeftRotatesLeft)
{
  {
    vector<int> input {sample_input};
    const vector<int> rotated {rotate_left(input, 1)};

    BOOST_TEST(rotated[0] == 2);
    BOOST_TEST(rotated[1] == 3);
    BOOST_TEST(rotated[2] == 4);
    BOOST_TEST(rotated[3] == 5);
    BOOST_TEST(rotated[4] == 1);
  }
  {
    vector<int> input {sample_input};
    const vector<int> rotated {rotate_left(input, 2)};

    BOOST_TEST(rotated[0] == 3);
    BOOST_TEST(rotated[1] == 4);
    BOOST_TEST(rotated[2] == 5);
    BOOST_TEST(rotated[3] == 1);
    BOOST_TEST(rotated[4] == 2);
  }
  {
    vector<int> input {sample_input};
    const vector<int> rotated {rotate_left(input, 3)};

    BOOST_TEST(rotated[0] == 4);
    BOOST_TEST(rotated[1] == 5);
    BOOST_TEST(rotated[2] == 1);
    BOOST_TEST(rotated[3] == 2);
    BOOST_TEST(rotated[4] == 3);
  }
  {
    // Test case 0
    // HackerRank
    vector<int> input {sample_input};
    const vector<int> rotated {rotate_left(input, 4)};

    BOOST_TEST(rotated[0] == 5);
    BOOST_TEST(rotated[1] == 1);
    BOOST_TEST(rotated[2] == 2);
    BOOST_TEST(rotated[3] == 3);
    BOOST_TEST(rotated[4] == 4);
  }
  {
    vector<int> input {sample_input};
    const vector<int> rotated {rotate_left(input, 5)};

    BOOST_TEST(rotated[0] == 1);
    BOOST_TEST(rotated[1] == 2);
    BOOST_TEST(rotated[2] == 3);
    BOOST_TEST(rotated[3] == 4);
    BOOST_TEST(rotated[4] == 5);
  }
  // Test case 1.
  // cf. https://www.hackerrank.com/challenges/ctci-array-left-rotation/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=arrays
  {
    vector<int> input {
      41, 73, 89, 7, 10, 1, 59, 58, 84, 77, 77, 97, 58, 1, 86, 58, 26, 10, 86,
      51};
    vector<int> output {
      77, 97, 58, 1, 86, 58, 26, 10, 86, 51, 41, 73, 89, 7, 10, 1, 59, 58, 84,
      77};

    const vector<int> rotated {rotate_left(input, 10)};
    BOOST_TEST(rotated == output);
  }

  // Test case 10.
  {
    vector<int> input {
      33, 47, 70, 37, 8, 53, 13, 93, 71, 72, 51, 100, 60, 87, 97};
    vector<int> output {
      87, 97, 33, 47, 70, 37, 8, 53, 13, 93, 71, 72, 51, 100, 60};

    const vector<int> rotated {rotate_left(input, 13)};
    BOOST_TEST(rotated == output);
  }
}

BOOST_AUTO_TEST_SUITE_END() // RotateLeft

BOOST_AUTO_TEST_SUITE(LeetCode)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(InplaceInsertionSortSortsSmallVectors)
{
  vector<int> arr {12, 11, 13, 5, 6};
  insertion_sort(arr);

  const vector<int> expected {5, 6, 11, 12, 13};
  BOOST_TEST(arr == expected);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CheckIfNAndItsDoubleExistUsingStdLibrary)
{
  {
    const vector<int> input {10, 2, 5, 3};
    BOOST_TEST(check_if_double_exists(input));
  }
  {
    const vector<int> input {7, 1, 14, 11};
    BOOST_TEST(check_if_double_exists(input));
  }
  {
    const vector<int> input {3, 1, 7, 11};
    BOOST_TEST(!check_if_double_exists(input));
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ValidMountainArrayDistinguishesValidMountains)
{
  {
    const vector<int> input;
    BOOST_TEST(!valid_mountain_array(input));
  }
  {
    const vector<int> input {2};
    BOOST_TEST(!valid_mountain_array(input));
  }
  {
    const vector<int> input {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    BOOST_TEST(!valid_mountain_array(input));
  }
  {
    const vector<int> input {0, 2, 3, 4, 5, 2, 1, 0};
    BOOST_TEST(valid_mountain_array(input));
  }
  {
    const vector<int> input {0, 2, 3, 3, 5, 2, 1, 0};
    BOOST_TEST(!valid_mountain_array(input));
  }
  {
    const vector<int> input {2, 1};
    BOOST_TEST(!valid_mountain_array(input));
  }
  {
    const vector<int> input {3, 5, 5};
    BOOST_TEST(!valid_mountain_array(input));
  }
  {
    const vector<int> input {0, 3, 2, 1};
    BOOST_TEST(valid_mountain_array(input));
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ReplaceWithGreatestOnRightReplacesWithGreatestInPlace)
{
  {
    vector<int> input {17, 18, 5, 4, 6, 1};
    const vector<int> expected {18, 6, 6, 6, 1, -1};
    BOOST_TEST(replace_with_greatest_on_right(input) == expected);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FastestReplaceWithGreatestOnRightCopiesToNewArray)
{
  {
    vector<int> input {17, 18, 5, 4, 6, 1};
    const vector<int> expected {18, 6, 6, 6, 1, -1};

    //const auto result = fastest_replace_with_greatest_on_right(input);

    BOOST_TEST(fastest_replace_with_greatest_on_right(input) == expected);
  }
}

// cf. https://leetcode.com/problems/median-of-two-sorted-arrays/description/
// 4. Median of Two Sorted Arrays
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FindSortedArraysMedianReturnsMedianOfTwoAssortedArrays)
{
  {
    vector<int> nums1 {1, 3};
    vector<int> nums2 {2};
    BOOST_TEST(find_sorted_arrays_median(nums1, nums2) == 2.0);
  }
  {
    vector<int> nums1 {1, 2};
    vector<int> nums2 {3, 4};
    BOOST_TEST(find_sorted_arrays_median(nums1, nums2) == 2.5);
  }
  {
    vector<int> nums1 {0, 0};
    vector<int> nums2 {0, 0};
    BOOST_TEST(find_sorted_arrays_median(nums1, nums2) == 0.0);
  }
  {
    vector<int> nums1;
    vector<int> nums2 {1};
    BOOST_TEST(find_sorted_arrays_median(nums1, nums2) == 1.0);
  }
  {
    vector<int> nums1 {2};
    vector<int> nums2;
    BOOST_TEST(find_sorted_arrays_median(nums1, nums2) == 2.0);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(
  FastestFindSortedArraysMedianReturnsMedianOfTwoAssortedArrays)
{
  {
    vector<int> nums1 {1, 3};
    vector<int> nums2 {2};
    BOOST_TEST(fastest_find_sorted_arrays_median(nums1, nums2) == 2.0);
  }
  {
    vector<int> nums1 {1, 2};
    vector<int> nums2 {3, 4};
    BOOST_TEST(fastest_find_sorted_arrays_median(nums1, nums2) == 2.5);
  }
  {
    vector<int> nums1 {0, 0};
    vector<int> nums2 {0, 0};
    BOOST_TEST(fastest_find_sorted_arrays_median(nums1, nums2) == 0.0);
  }
  {
    vector<int> nums1;
    vector<int> nums2 {1};
    BOOST_TEST(fastest_find_sorted_arrays_median(nums1, nums2) == 1.0);
  }
  {
    vector<int> nums1 {2};
    vector<int> nums2;
    BOOST_TEST(fastest_find_sorted_arrays_median(nums1, nums2) == 2.0);
  }
}


BOOST_AUTO_TEST_SUITE_END() // LeetCode

BOOST_AUTO_TEST_SUITE_END() // Arrays_tests
BOOST_AUTO_TEST_SUITE_END() // DataStructures