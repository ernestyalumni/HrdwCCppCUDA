#include "DataStructures/Arrays/ArrayQuestions.h"

#include <boost/test/unit_test.hpp>
#include <string>
#include <vector>

using
  DataStructures::Arrays::ArrayQuestions::CrackingTheCodingInterview::
    is_unique_character_string;
using DataStructures::Arrays::ArrayQuestions::LeetCode::duplicate_zeros;
using DataStructures::Arrays::ArrayQuestions::LeetCode::
  duplicate_zeros_linear_time;
using DataStructures::Arrays::ArrayQuestions::LeetCode::
  duplicate_zeros_with_shift;
using DataStructures::Arrays::ArrayQuestions::LeetCode::
  find_even_length_numbers;
using DataStructures::Arrays::ArrayQuestions::LeetCode::
  find_max_consecutive_ones;
using DataStructures::Arrays::ArrayQuestions::LeetCode::max_profit;
using DataStructures::Arrays::ArrayQuestions::LeetCode::merge_sorted_arrays;
using DataStructures::Arrays::ArrayQuestions::LeetCode::remove_element;
using DataStructures::Arrays::ArrayQuestions::LeetCode::sorted_squares;
using DataStructures::Arrays::ArrayQuestions::LeetCode::sorted_squares_two_ptrs;
using std::string;
using std::vector;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Arrays)
BOOST_AUTO_TEST_SUITE(ArrayQuestions_tests)

//------------------------------------------------------------------------------
/// \ref https://en.cppreference.com/w/cpp/language/ascii
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ASCIIDecimalConversionToChar)
{
  int i {33};

  BOOST_TEST(static_cast<char>(i) == '!');
}

BOOST_AUTO_TEST_CASE(CharConversionToASCIIDecimal)
{
  BOOST_TEST(static_cast<int>('#') == 35);

  const char c {'('};

  BOOST_TEST(static_cast<int>(c) == 40);
}

//------------------------------------------------------------------------------
/// \ref Gayle Laakmann McDowell, 6th Ed.
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_SUITE(CrackingTheCodingInterview_tests)

//------------------------------------------------------------------------------
/// \ref pp. 90, 1.1 Is Unique, McDowell, 6th. Ed.
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IsUniqueCharacterStringWorksWithCArray)
{
  const string duplicate_string {"GeeksforGeeks"};
  const string not_duplicate_string {"algorithm"};

  BOOST_TEST(!is_unique_character_string(duplicate_string));
  BOOST_TEST(is_unique_character_string(not_duplicate_string));
}

BOOST_AUTO_TEST_SUITE_END() // CrackingTheCodingInterview_tests

BOOST_AUTO_TEST_SUITE(Leetcode_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BestTimeToBuyAndSellStockIIElementaryCases)
{
  {
    vector<int> prices {42};
    BOOST_TEST(max_profit(prices) == 0);
  }
  {
    vector<int> prices {7, 2};
    BOOST_TEST(max_profit(prices) == 0);
  }
  {
    vector<int> prices {3, 5};
    BOOST_TEST(max_profit(prices) == 2);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BestTimeToBuyAndSellStockIIWithPointers)
{
  {
    const vector<int> prices {7, 1, 5, 3, 6, 4};

  }
  {
    const vector<int> prices {1, 2, 3, 4, 5};
    
  }
  {
    const vector<int> prices {7, 6, 4, 3, 1};
    
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FindMaxConsecutiveOnesFindsMaxLength)
{
  vector<int> example_1 {1, 1, 0, 1, 1, 1};

  BOOST_TEST(find_max_consecutive_ones(example_1) == 3);

  vector<int> run_code_example {1, 0, 1, 1, 0, 1};
  BOOST_TEST(find_max_consecutive_ones(run_code_example) == 2);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FindEvenLengthNumbersReturnsCorrectTotal)
{
  vector<int> nums1 {12, 345, 2, 6, 7896};

  vector<int> nums2 {555, 901, 482, 1771};

  BOOST_TEST(find_even_length_numbers(nums1) == 2);
  BOOST_TEST(find_even_length_numbers(nums2) == 1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SortedSquaresSortsSquareOfArrayInPlace)
{
  vector<int> example_1_input {-4, -1, 0, 3, 10};
  vector<int> example_2_input {-7, -3, 2, 3, 11};

  BOOST_TEST((sorted_squares(example_1_input) ==
    vector<int>{0, 1, 9, 16, 100}));
  BOOST_TEST((sorted_squares(example_2_input) ==
    vector<int>{4, 9, 9, 49, 121}));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SortedSquaresTwoPtrsSortsSquareOfArrayInPlace)
{
  vector<int> example_1_input {-4, -1, 0, 3, 10};
  vector<int> example_2_input {-7, -3, 2, 3, 11};

  BOOST_TEST((sorted_squares_two_ptrs(example_1_input) ==
    vector<int>{0, 1, 9, 16, 100}));
  BOOST_TEST((sorted_squares_two_ptrs(example_2_input) ==
    vector<int>{4, 9, 9, 49, 121}));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DuplicateZerosShiftsToRightAndDuplicateZeros)
{
  vector<int> example_1_input {1, 0, 2, 3, 0, 4, 5, 0};
  vector<int> example_2_input {1, 2, 3};

  duplicate_zeros(example_1_input);
  duplicate_zeros(example_2_input);

  BOOST_TEST((example_1_input == vector<int>{1, 0, 0, 2, 3, 0, 0, 4}));
  BOOST_TEST((example_2_input == vector<int>{1, 2, 3}));

  vector<int> test_case_a {0, 4, 1, 0, 0, 8, 0, 0, 3};
  duplicate_zeros(test_case_a);
  BOOST_TEST((test_case_a == vector<int>{0, 0, 4, 1, 0, 0, 0, 0, 8}));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DuplicateZerosLinearTimeShiftsToRightAndDuplicateZeros)
{
  vector<int> example_1_input {1, 0, 2, 3, 0, 4, 5, 0};
  vector<int> example_2_input {1, 2, 3};

  duplicate_zeros_linear_time(example_1_input);
  duplicate_zeros_linear_time(example_2_input);

  BOOST_TEST((example_1_input == vector<int>{1, 0, 0, 2, 3, 0, 0, 4}));
  BOOST_TEST((example_2_input == vector<int>{1, 2, 3}));

  vector<int> test_case_a {0, 4, 1, 0, 0, 8, 0, 0, 3};
  duplicate_zeros_linear_time(test_case_a);
  BOOST_TEST((test_case_a == vector<int>{0, 0, 4, 1, 0, 0, 0, 0, 8}));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DuplicateZerosWithShiftShiftsToRightAndDuplicateZeros)
{
  vector<int> example_1_input {1, 0, 2, 3, 0, 4, 5, 0};
  vector<int> example_2_input {1, 2, 3};

  duplicate_zeros_with_shift(example_1_input);
  duplicate_zeros_with_shift(example_2_input);

  BOOST_TEST((example_1_input == vector<int>{1, 0, 0, 2, 3, 0, 0, 4, 5, 0, 0}));
  BOOST_TEST((example_2_input == vector<int>{1, 2, 3}));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MergeSortedArraysFromReverse)
{
  vector<int> example_nums1 {1, 2, 3, 0, 0, 0};
  vector<int> example_nums2 {2, 5, 6};

  merge_sorted_arrays(example_nums1, 3, example_nums2, 3);

  BOOST_TEST((example_nums1 == vector<int>{1, 2, 2, 3, 5, 6}));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(RemoveElementRemovesElementFromArray)
{
  vector<int> example_1_nums {3, 2, 2, 3};
  vector<int> example_2_nums {0, 1, 2, 2, 3, 0, 4, 2};

  BOOST_TEST(remove_element(example_1_nums, 3) == 2);
  BOOST_TEST(remove_element(example_2_nums, 2) == 5);

}

BOOST_AUTO_TEST_SUITE_END() // Leetcode_tests

BOOST_AUTO_TEST_SUITE_END() // ArrayQuestions_tests
BOOST_AUTO_TEST_SUITE_END() // Arrays
BOOST_AUTO_TEST_SUITE_END() // DataStructures