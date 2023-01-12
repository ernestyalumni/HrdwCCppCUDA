//------------------------------------------------------------------------------
/// \author Ernest Yeung
/// \brief Array type Questions.
//-----------------------------------------------------------------------------
#include "ArrayQuestions.h"

#include <cstring> // std::memset
#include <utility> // std::swap, also in <algorithm>
#include <vector>

using std::swap;
using std::vector;

namespace DataStructures
{
namespace Arrays
{
namespace ArrayQuestions
{

namespace CrackingTheCodingInterview
{

bool is_unique_character_string(const std::string& s)
{
  constexpr std::size_t N {256};

  bool first_seen[N];

  std::memset(first_seen, static_cast<int>(false), 256);

  // O(|s|) time complexity.
  for (char c : s)
  {
    const int ascii_decimal {static_cast<int>(c)};

    if (first_seen[ascii_decimal])
    {
      return false;
    }
    else
    {
      first_seen[ascii_decimal] = true;
    }
  }

  return true;
}

} // namespace CrackingTheCodingInterview

namespace LeetCode
{

int max_profit(vector<int>& prices)
{
  const int N {static_cast<int>(prices.size())};

  if (N < 2)
  {
    return 0;
  }

  if (N == 2)
  {
    return prices.at(1) > prices.at(0) ? prices.at(1) - prices.at(0) : 0;
  }

  int current_buy_ptr {N - 2};
  int current_sale_ptr {N - 1};
  int current_max_profit {
    prices.at(current_sale_ptr) > prices.at(current_buy_ptr) ? 
      prices.at(current_sale_ptr) - prices.at(current_buy_ptr) : 0};


  return 0;
}

//------------------------------------------------------------------------------
/// \details Runtime: 76 ms
/// Memory Usage: 36.5 MB  
/// \url https://leetcode.com/submissions/detail/414815033/?from=/explore/learn/card/fun-with-arrays/521/introduction/3238/
//------------------------------------------------------------------------------
int find_max_consecutive_ones(vector<int>& nums)
{
  int global_max_ones_length {0};
  int counter {0};

  for (auto iter {nums.begin()}; iter != nums.end(); ++iter)
  {
    if (*iter == 1)
    {
      counter += 1;
    }
    // *iter == 0
    else
    {
      const int new_max {counter};
      counter = 0;
      if (new_max > global_max_ones_length)
      {
        global_max_ones_length = new_max;
      } 
    }
  }

  // We've reached the end, but deal with case where we didn't terminate
  // a sequence of 1s.
  if (global_max_ones_length < counter)
  {
    global_max_ones_length = counter;
  }
  return global_max_ones_length;
}

int find_even_length_numbers(vector<int>& nums)
{
  auto is_even_length = [](int num)
  {
    bool is_even {true};
    int dividend {num};

    while (dividend != 0)
    {
      dividend = dividend / 10;
      if (!is_even)
      {
        is_even = true;
      }
      else
      {
        is_even = false;
      }
    }
    return is_even;
  };

  int total_even_length_numbers {0};

  for (auto num : nums)
  {
    if (is_even_length(num))
    {
      ++total_even_length_numbers;
    }
  }
  return total_even_length_numbers;
}

vector<int> sorted_squares(vector<int>& A)
{
  int non_negative_start_index {0};
  for (auto a : A)
  {
    if (a >= 0)
    {
      break;
    }

    ++non_negative_start_index;
  }

  for (auto& a  : A)
  {
    a = a * a;
  }

  // Do insertion sort.
  /*for (int index {0}; index < non_negative_start_index; ++index)
  {
    const int target_value {a.at(index)};

    for (int swap_index {non_negative_start_index};
      swap_index < A.size();
      ++swap_index)
    {

    }
  }
  */

  for (int index {non_negative_start_index}; index > 0; --index)
  {
    int j {index - 1};

    while (j < (A.size() - 1) && A.at(j) > A.at(j + 1))
    {
      swap(A[j], A[j + 1]);
      ++j;
    }
  }

  return A;
}

//-----------------------------------------------------------------------------
/// \url https://leetcode.com/problems/squares-of-a-sorted-array/solution/
/// \brief Two pointer approach.
/// \details Strategy: iterate over negative part in reverse, and positive part
/// in forward direction.
//-----------------------------------------------------------------------------
vector<int> sorted_squares_two_ptrs(vector<int>& A)
{
  // l < r. 0 <= l <= N - 1 since all numbers can be negative.
  size_t l {0};
  // Possibly, 1 <= r <= N
  size_t r {A.size()};
  // Go in reverse: 0 <= p <= N - 1
  size_t p {A.size() - 1};

  vector<int> results (A.size(), 0);

  while (l < r)
  {
    int A_l_sq {A.at(l) * A.at(l)};
    int A_rm1_sq {A.at(r - 1) * A.at(r - 1)};

    if (A_l_sq >= A_rm1_sq)
    {
      results.at(p) = A_l_sq;
      --p;
      ++l;
    }
    else
    {
      results.at(p) = A_rm1_sq;
      --p;
      --r;
    }
  }

  return results;
}

//------------------------------------------------------------------------------
/// \url https://leetcode.com/explore/learn/card/fun-with-arrays/525/inserting-items-into-an-array/3245/
/// \brief Duplicate Zeros.
/// \details Given fixed length array arr of integers, duplicate each occurrence
/// of zero, shifting remaining elements to right.
//------------------------------------------------------------------------------
void duplicate_zeros(std::vector<int>& arr)
{
  /*
  for (size_t i {0}; i < arr.size(); ++i)
  {
    size_t reverse_i {arr.size() - 1 - i};

    if (arr.at(reverse_i) == 0)
    {
      // Shift elements to right starting with index reverse_i - 1
      for (size_t k {0}; k < i; ++k)
      {
        size_t reverse_k {arr.size() - 1 - k};

        arr.at(reverse_k) = arr.at(reverse_k - 1);
      }

      // Move i further, up 1 more spot to not duplicate the newly duplicated 0.
      ++i;
    }
  }
  */

  size_t i {0};

  while (i < arr.size())
  {
    if (arr.at(i) == 0)
    {
      for (size_t k {i}; k < arr.size() - 1; ++k)
      {
        // Go in reverse in order not to overwrite any entries prior.
        size_t reverse_k {arr.size() - 1 - k + i};

        arr.at(reverse_k) = arr.at(reverse_k - 1);
      }

      // Move forward-moving index up 2 to jump over newly duplicated 0.
      i += 2; 
    }
    // No 0 found; keep going incrementally.
    else
    {
      ++i;
    }
  }
}

//------------------------------------------------------------------------------
/// \date Oct 30, 03:30 Start.
/// Oct 30, 04:29 finished, understood subtle if statement.
//------------------------------------------------------------------------------
void duplicate_zeros_linear_time(vector<int>& arr)
{
  // Count number of zeros in order to know how many spaces to shift elements by
  // because the final array will have elements shifted by that net number of
  // zeros.
  const int N {static_cast<int>(arr.size())};

  int total_number_of_zeros {0};
  for (int i {0}; i < arr.size(); ++i)
  {
    if (arr.at(i) == 0)
    {
      total_number_of_zeros++;
    }
  }

  if (total_number_of_zeros == 0)
  {
    return;
  }

  // r will iterate over r = N - 1, ... total_number_of_zeros, and represents
  // the position of where to put the value.
  int current_index {N - 1};
  int r {current_index + total_number_of_zeros};

  while (current_index >= 0 && r >= 0)
  {
    if (arr.at(current_index) != 0)
    {
      if (r < N)
      {
        arr.at(r) = arr.at(current_index);
      }
    }
    else
    {
      if (r < N)
      {
        // Set r to zero.
        arr.at(r) = arr.at(current_index);
      }
      r--;
      if (r < N)
      {
        arr.at(r) = arr.at(current_index);
      }
    }
    r--;
    current_index--;
  }
}

void duplicate_zeros_with_shift(std::vector<int>& arr)
{
  size_t l {0};
  size_t N {arr.size()};

  while (l < N)
  {
    if (arr.at(l) == 0)
    {
      // index = l, l + 1, ... N - 1, N - 1 - l items
      // where l = 0, 1, ... N -1
      for (size_t index {l}; index < N; ++index)
      {
        // N - 1, ... N - 2, ... l
        size_t reverse_index {N - 1 - index + l};

        if (reverse_index == N - 1)
        {
          arr.emplace_back(arr.at(reverse_index));
        }
        // reverse_index < N - 1
        else
        {
          arr.at(reverse_index + 1) = arr.at(reverse_index);
        }
      } // Finished shifting to the right.
      // Account for the increase in size by 1.
      ++N;
      // Skip over the duplicate 0.
      l += 2;
    }
    else
    {
      // Advance to next value.
      ++l;
    }
  }
}

//------------------------------------------------------------------------------
/// \url https://leetcode.com/explore/learn/card/fun-with-arrays/525/inserting-items-into-an-array/3253/
/// \brief Merge Sorted Array.
/// \details Given 2 sorted integer arrays nums1 and nums2, merge nums2 into
/// nums2 as 1 sorted array.
/// \date Oct 30, 07:46, Done 08:23.
/// Runtime: 0 ms
/// Memory Usage: 9.6 MB
//------------------------------------------------------------------------------
void merge_sorted_arrays(
  std::vector<int>& nums1,
  int m,
  std::vector<int>& nums2,
  int n)
{
  // Right most index on the total array nums1.
  int r {m + n - 1};
  // These indices decrement from the "ends" of nums1, nums2 respectively.
  int reverse_index_1 {m - 1};
  int reverse_index_2 {n - 1};

  while (r >= 0 && reverse_index_1 >= 0 && reverse_index_2 >= 0)
  {
    if (nums1.at(reverse_index_1) > nums2.at(reverse_index_2))
    {
      nums1.at(r) = nums1.at(reverse_index_1);
      --reverse_index_1;
    }
    else
    {
      nums1.at(r) = nums2.at(reverse_index_2);
      --reverse_index_2;
    }

    --r;
  }

  if (reverse_index_1 < 0)
  {
    while (r >= 0 && reverse_index_2 >= 0)
    {
      nums1.at(r) = nums2.at(reverse_index_2);
      --r;
      --reverse_index_2;
    }
  }
  // Otherwise nums1 already sorted.
/*  else if (reverse_index_2 < 0)
  {
    while (r >= 0 && reverse_index_1 >= 0)
    {
      nums1.at(r) = nums2.at(reverse_index_1);
      --r;
      --reverse_index_1;
    }
  }*/
}

//------------------------------------------------------------------------------
/// \details Similar solution to Remove Duplicates from Sorted Array.
//------------------------------------------------------------------------------

int remove_element(vector<int>& nums, int val)
{
  /*
  if (nums.size() == 0)
  {
    return 0;
  }
  else if (nums.size() == 1)
  {
    if 
  }


  // Place removed elements in this position.
  size_t r {nums.size() - 1};
  size_t current_index {0};

  while (current_index < r)
  {
    if (nums.at(current_index) == val)
    {
      swap(nums.at(current_index), nums.at(r));
      --r;
    }
    else
    {
      ++current_index;
    }
  }

  return static_cast<int>(current_index + 1);
  */

  // https://leetcode.com/problems/remove-element/solution/

  size_t current_index {0};

  for (size_t j {0}; j < nums.size(); ++j)
  {
    if (nums.at(j) != val)
    {
      nums.at(current_index) = nums.at(j);
      ++current_index;
    }
  }

  return current_index;
}

} // namespace LeetCode

} // namespace ArrayQuestions
} // namespace Arrays
} // namespace DataStructures
