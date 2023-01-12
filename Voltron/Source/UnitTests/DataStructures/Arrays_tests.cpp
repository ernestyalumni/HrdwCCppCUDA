//------------------------------------------------------------------------------
/// \file Arrays_tests.cpp
/// \date 20201015 15, 17:21 complete
//------------------------------------------------------------------------------
#include "DataStructures/Arrays.h"

#include <boost/test/unit_test.hpp>
#include <cstddef> // std::size_t
#include <iterator>
#include <vector>

using DataStructures::Arrays::Array;
using DataStructures::Arrays::LeetCode::check_if_double_exists;
using DataStructures::Arrays::LeetCode::fastest_find_sorted_arrays_median;
using DataStructures::Arrays::LeetCode::fastest_replace_with_greatest_on_right;
using DataStructures::Arrays::LeetCode::find_sorted_arrays_median;
using DataStructures::Arrays::LeetCode::insertion_sort;
using DataStructures::Arrays::LeetCode::remove_duplicates;
using DataStructures::Arrays::LeetCode::replace_with_greatest_on_right;
using DataStructures::Arrays::LeetCode::valid_mountain_array;
using DataStructures::Arrays::rotate_left;
using std::begin;
using std::size_t;
using std::vector;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Arrays_tests)

BOOST_AUTO_TEST_SUITE(Array_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AllocationAndDeallocationOfNewArrays)
{
  size_t capacity {2};
  int* items {new int[capacity]};
  items[0] = 11;
  items[1] = 22;
  BOOST_TEST(items[0] == 11);
  BOOST_TEST(items[1] == 22);

  int* new_items {new int[capacity * 2]};

  for (int index {0}; index < capacity; ++index)
  {
    new_items[index] = items[index];
  }

  BOOST_TEST(new_items[0] == 11);
  BOOST_TEST(new_items[1] == 22);
  new_items[2] = 33;
  new_items[3] = 44;
  BOOST_TEST(new_items[2] == 33);
  BOOST_TEST(new_items[3] == 44);

  delete[] items;

  items = new_items;

  BOOST_TEST(items[0] == 11);
  BOOST_TEST(items[1] == 22);
  BOOST_TEST(items[2] == 33);
  BOOST_TEST(items[3] == 44);

  delete[] items;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructArrayWithSize)
{
  Array<int> a {5};

  BOOST_TEST(a.length() == 0);
  BOOST_TEST(a.capacity() == 5);

  for (int i {0}; i < a.capacity(); ++i)
  {
    a.insert(i * 11, i);
    BOOST_TEST(a.length() == i + 1);
  }

  BOOST_TEST(a.length() == a.capacity());

  for (int i {0}; i < a.capacity(); ++i)
  {
    BOOST_TEST(a[i] == i * 11);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AddAndDeleteElementsAndCheck)
{
  Array<int> a {7};
  BOOST_TEST(a.length() == 0);
  BOOST_TEST(a.capacity() == 7);

  for (int i {0}; i < a.capacity(); ++i)
  {
    a.insert(i * 11, i);
    BOOST_TEST(a.length() == i + 1);
  }

  BOOST_TEST(a.length() == a.capacity());

  a.delete_at_index(3);
  BOOST_TEST(a.length() == a.capacity() - 1);
  for (int i {0}; i < 3; ++i)
  {
    BOOST_TEST(a[i] == i * 11);
  }
  for (int i {3}; i < a.length(); ++i)
  {
    BOOST_TEST(a[i] == ((i + 1) * 11));
  }
}

BOOST_AUTO_TEST_SUITE_END() // Array_tests

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

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(RemoveDuplicatesWithTwoPointers)
{
  vector<int> example_1_nums {1, 1, 2};
  vector<int> example_2_nums {0, 0, 1, 1, 1, 2, 2, 3, 3, 4};

  BOOST_TEST(remove_duplicates(example_1_nums) == 2);
  BOOST_TEST((example_1_nums == vector<int>{1, 2, 2}));

  BOOST_TEST(remove_duplicates(example_2_nums) == 5);
  BOOST_TEST((example_2_nums == vector<int>{0, 1, 2, 3, 4, 2, 2, 3, 3, 4}));

}

BOOST_AUTO_TEST_SUITE_END() // LeetCode

BOOST_AUTO_TEST_SUITE_END() // Arrays_tests
BOOST_AUTO_TEST_SUITE_END() // DataStructures