//------------------------------------------------------------------------------
/// \author Ernest Yeung
/// \brief Array type questions.
//------------------------------------------------------------------------------
#ifndef DATA_STRUCTURES_ARRAYS_ARRAY_QUESTIONS_H
#define DATA_STRUCTURES_ARRAYS_ARRAY_QUESTIONS_H

#include <string>
#include <vector>

namespace DataStructures
{
namespace Arrays
{
namespace ArrayQuestions
{

namespace CrackingTheCodingInterview
{

//------------------------------------------------------------------------------
/// \details Time complexity O(s) <= O(255), Size complexity
/// (256 * sizeof(bool))
/// \ref https://stackoverflow.com/a/4987875
//------------------------------------------------------------------------------
bool is_unique_character_string(const std::string& s);

} // namespace CrackingTheCodingInterview

namespace LeetCode
{

int max_profit(std::vector<int>& prices);

//------------------------------------------------------------------------------
/// \name Max Consecutive Ones
/// \url https://leetcode.com/explore/learn/card/fun-with-arrays/521/introduction/3238/
/// \brief Given a binary array, find the maximum number of consecutive 1s in
/// this array.
//------------------------------------------------------------------------------
int find_max_consecutive_ones(std::vector<int>& nums);

//------------------------------------------------------------------------------
/// \name Find Numbers with Even Number of Digits
/// \url https://leetcode.com/explore/learn/card/fun-with-arrays/521/introduction/3240/
/// \brief Given an array of integers A sorted in non-decreasing order, return
/// an array of squares of each number, also in sorted non-decreasing order.
//------------------------------------------------------------------------------
int find_even_length_numbers(std::vector<int>& nums);

//------------------------------------------------------------------------------
/// \name Squares of a Sorted Array
/// \url https://leetcode.com/explore/learn/card/fun-with-arrays/521/introduction/3237/
/// \brief Given an array nums of integers, return how many of them contain an
/// even number of digits.
//------------------------------------------------------------------------------
std::vector<int> sorted_squares(std::vector<int>& A);

//------------------------------------------------------------------------------
/// \url https://leetcode.com/problems/squares-of-a-sorted-array/solution/
/// \brief Two pointer approach.
/// \details Strategy: iterate over negative part in reverse, and positive part
/// in forward direction. Then build a merged array from two sorted arrays by
/// comparing element by element.
//------------------------------------------------------------------------------
std::vector<int> sorted_squares_two_ptrs(std::vector<int>& A);

//------------------------------------------------------------------------------
/// \url https://leetcode.com/explore/learn/card/fun-with-arrays/525/inserting-items-into-an-array/3245/
/// \brief Duplicate Zeros.
/// \details Given fixed length array arr of integers, duplicate each occurrence
/// of zero, shifting remaining elements to right.
//------------------------------------------------------------------------------
void duplicate_zeros(std::vector<int>& arr);

//------------------------------------------------------------------------------
/// \date Oct 30, 03:30
//------------------------------------------------------------------------------
void duplicate_zeros_linear_time(std::vector<int>& arr);

void duplicate_zeros_with_shift(std::vector<int>& arr);

//------------------------------------------------------------------------------
/// \url https://leetcode.com/explore/learn/card/fun-with-arrays/525/inserting-items-into-an-array/3253/
/// \brief Merge Sorted Array.
/// \details Given 2 sorted integer arrays nums1 and nums2, merge nums2 into
/// nums2 as 1 sorted array.
/// \date Oct 30, 07:46
//------------------------------------------------------------------------------
void merge_sorted_arrays(
  std::vector<int>& nums1,
  int m,
  std::vector<int>& nums2,
  int n);

//------------------------------------------------------------------------------
/// \url https://leetcode.com/explore/learn/card/fun-with-arrays/526/deleting-items-from-an-array/3247/
/// \brief Remove Element.
//------------------------------------------------------------------------------
int remove_element(std::vector<int>& nums, int val);

} // namespace LeetCode

} // namespace ArrayQuestions
} // namespace Arrays
} // namespace DataStructures

#endif // DATA_STRUCTURES_ARRAYS_ARRAY_QUESTIONS_H