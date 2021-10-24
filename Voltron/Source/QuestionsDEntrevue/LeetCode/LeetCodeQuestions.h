//------------------------------------------------------------------------------
/// \file LeetCodeQuestions.h
/// \author Ernest Yeung
/// \brief 
///
/// \ref https://leetcode.com/explore/learn/card/data-structure-tree/134/traverse-a-tree/992/
/// \ref LeetCode, Binary Trees
//-----------------------------------------------------------------------------
#ifndef QUESTIONS_D_ENTREVUE_LEETCODE_LEETCODE_QUESTIONS_H
#define QUESTIONS_D_ENTREVUE_LEETCODE_LEETCODE_QUESTIONS_H

#include <deque>
#include <iterator> // std::distance
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <utility> // std::make_pair, std::pair
#include <vector>

namespace QuestionsDEntrevue
{

namespace LeetCode
{

//------------------------------------------------------------------------------
/// \url https://leetcode.com/problems/valid-parentheses/
/// \name 20. Valid Parentheses. Easy.
/// \brief Given a string s containing just the characters '(', ')', '{', '}',
/// and '[', ']', determine if input string is valid.
///
/// \details Use stack.
//------------------------------------------------------------------------------
bool is_valid_parentheses(std::string s);

//------------------------------------------------------------------------------
/// \url https://leetcode.com/problems/longest-valid-parentheses/
/// \name 32. Longest Valid Parentheses. Hard.
/// \brief Given a string containing just the characters '(' and ')', find the
/// length of the longest valid (well-formed) parentheses substring.
///
/// Use the fact that we keep track of indices in the stack and that we can
/// always access the value on the string through the index.
//------------------------------------------------------------------------------
int longest_valid_parentheses(std::string s);

//------------------------------------------------------------------------------
/// \url https://leetcode.com/problems/maximum-subarray/
/// \name 53. Maximum Subarray.
/// \brief Given an integer array nums, find the contiguous subarray (containing
/// at least one number) which has the largest sum and return its sum.
///
/// \details Brute force is O(N^3) or O(N^2) as you check every subarray
/// starting from I = 0, 1, ... N - 1, and ending at J = 0, 1, ... N- 1.
/// For all linear array problems, aim to solve in O(N) linear time.
/// Don't think of tables, think of what is the subproblem?
/// Key observation, at an element and if at that element, J is index of that
/// element, what is the max contiguous subarray?
///
/// Other key observation is that max contiguous subarray for entire array must
/// end on a element in the array. That's why we can ask the subproblem, what's
/// the max contiguous subarray with condition that it ends at index J.
///
/// \url https://youtu.be/2MmGzdiKR9Y
///
/// Easy.
//------------------------------------------------------------------------------
int max_subarray(std::vector<int>& nums);

//------------------------------------------------------------------------------
/// \url https://leetcode.com/problems/climbing-stairs/description/
/// \name 70. Climbing Stairs.
/// \brief You are climbing a stair case. It takes n steps to reach the top.
///
/// \details Each time you can either climb 1 or 2 steps.
//------------------------------------------------------------------------------
int climb_stairs(int n);

int climb_stairs_iterative(const int n);

//------------------------------------------------------------------------------
/// \url https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/
/// \name 121. Best Time to Buy and Sell Stock.
/// \brief You are climbing a stair case. It takes n steps to reach the top.
///
/// \details Each time you can either climb 1 or 2 steps.
///
/// Traverse from left to get minimum buy prices dependent on time.
/// Traverse from right to get maximum sell prices dependent on time.
/// Traverse all 3 to get maximum profit at any time t. Then take max.
//------------------------------------------------------------------------------
int max_profit(std::vector<int>& prices);

//------------------------------------------------------------------------------
/// \url https://leetcode.com/problems/coin-change/
/// \name 322. Coin Change
/// \brief You are given coins of different denominations and a total amount of
/// money amount. Write a function to compute the fewest number of coins that
/// you need to make up that amount.
/// \details If that amount of money cannot be made up by any combination of the
/// coins, return -1.
///
/// Medium.
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
/// \details Iterative approach.
/// 1 <= coins.length <= 12
//------------------------------------------------------------------------------
int coin_change_bottom_up(std::vector<int>& coins, int amount);

int coin_change_top_down(
  std::vector<int>& coins,
  int amount);

int coin_change_top_down_step(
  std::vector<int>& coins,
  int amount,
  std::map<int, int>& min_coins);

//------------------------------------------------------------------------------
/// \details Recursive approach.
//------------------------------------------------------------------------------
int min_coin_change_recursive_step(
  std::vector<int>& coins,
  const int smallest_coinage,
  std::vector<std::optional<int>>& min_coins_for_amount,
  int amount);

int coin_change_recursive(std::vector<int>& coins, int amount);

//------------------------------------------------------------------------------
/// \url https://leetcode.com/problems/max-sum-of-rectangle-no-larger-than-k/
/// \name 363. Max Sum of Rectangle No Larger Than K.
/// \ref https://www.youtube.com/watch?v=-FgseNO-6Gk
/// Back To Back SWE, Maximum Sum Rectangle In A 2D Matrix - Kadane's Algorithm
//------------------------------------------------------------------------------
int max_sum_submatrix(std::vector<std::vector<int>>& matrix);

std::pair<int, std::pair<std::size_t, std::size_t>> find_subrow_max(
  std::vector<int>& row);

//------------------------------------------------------------------------------
/// \url https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/
/// \name 647. Palindromic Substrings.
/// \brief Given a string, your task is to count how many palindromic substrings
/// in this string.
/// \details Consider using multiple variables, pointers.
///
/// Runtime: 12 ms, faster than 60.36% of C++ online submissions for Palindromic
/// Substrings.
/// Memory Usage: 6.7 MB, less than 5.03% of C++ online submissions for
/// Palindromic Substrings.
///
/// cf. https://youtu.be/r1MXwyiGi_U
/// Top 10 Algorithms for the Coding Interview (for software engineers),
/// TechLead, Jun 27, 2019.
/// 1. DFS, 2. BFS, 3. Matching Parentheses, 4. Hash Tables, 5. Variables, ptrs
/// manipulation, 6. reversing linked list, 7. sorting fundamentals (quick,
/// merge, bubble sort, runtime of sort, time/space complexity), 8. recursion,
/// 9. custom data structures (OOP), 10. Binary search.
/// https://www.youtube.com/watch?v=r1MXwyiGi_U&lc=UgwGnloT3M405KnWQnd4AaABAg
//------------------------------------------------------------------------------
int count_palindromic_substrings(std::string s);

//------------------------------------------------------------------------------
/// \details Given length l, number of operations (comparisons) is l / 2 in
/// worst case (worse case is that it's a palindrome).
//------------------------------------------------------------------------------
inline auto check_palindrome = [](auto start_iter, auto end_iter) -> bool
{
  bool is_palindrome {true};

  // std::distance is the number of hops from start_iter to end_iter.
  while (std::distance(start_iter, end_iter) >= 0)
  {
    if (*start_iter == *end_iter)
    {
      ++start_iter;
      --end_iter;
    }
    else
    {
      is_palindrome = false;
      break;
    }
  }

  return is_palindrome;
};

inline auto find_even_size_palindromes = [](
  auto start_limit,
  auto end_limit,
  auto start_iter) -> int
{
  int count {0};

  auto end_iter = start_iter + 1;

  while (end_iter != end_limit && std::distance(start_limit, start_iter) >= 0)
  {
    if (*start_iter == *end_iter)
    {
      ++count;
      ++end_iter;
      --start_iter;
    }
    else
    {
      break;
    }
  }

  return count;
};

inline auto find_odd_size_palindromes = [](
  auto start_limit,
  auto end_limit,
  auto start_iter)
{
  // Count a single letter to be a palindrome itself.
  int count {1};

  auto end_iter = start_iter + 1;
  --start_iter;  

  while (end_iter != end_limit && std::distance(start_limit, start_iter) >= 0)
  {
    if (*start_iter == *end_iter)
    {
      ++count;
      ++end_iter;
      --start_iter;
    }
    else
    {
      break;
    }
  }

  return count;
};

//------------------------------------------------------------------------------
/// \url https://leetcode.com/problems/palindromic-substrings/solution/
/// \brief Expand around center.
/// Let N be the length of the string. The middle of the palindrome could be in
/// 1 of 2N-1 total positions: either at the letter, or in between 2 letters.
/// Thus, count the number of letters, N, and number of positions in between 2
/// letters, N - 1 -> N + N - 1 = 2N - 1
//------------------------------------------------------------------------------
int count_palindromic_substrings_simple(std::string s);

} // namespace LeetCode
} // namespace QuestionsDEntrevue

#endif // QUESTIONS_D_ENTREVUE_LEETCODE_LEETCODE_QUESTIONS_H