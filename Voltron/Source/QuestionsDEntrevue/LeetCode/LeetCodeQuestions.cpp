//------------------------------------------------------------------------------
/// \file LeetCodeQuestions.cpp
/// \author Ernest Yeung
/// \brief 
///
/// \ref https://leetcode.com/explore/learn/card/data-structure-tree/134/traverse-a-tree/992/
/// \ref LeetCode, Binary Trees
//-----------------------------------------------------------------------------
#include "LeetCodeQuestions.h"

#include <algorithm> // std::max, std::max_element, std::min_element, std::min
#include <array>
#include <cstddef> // std::size_t
#include <deque>
#include <iterator>
#include <map>
#include <optional> // std::make_optional
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility> // std::make_pair, std::pair, std::move;
#include <vector>
#include <stack>

using std::array;
using std::distance;
using std::find; // in <algorithm>
using std::make_optional;
using std::make_pair;
using std::map;
using std::max;
using std::max_element;
using std::min;
using std::min_element;
using std::move;
using std::nullopt;
using std::optional;
using std::pair;
using std::plus; // <algorithm>, function opbject effectively calls operator+
using std::size_t;
using std::stack;
using std::string;
using std::transform;
using std::unordered_map;
using std::vector;

namespace QuestionsDEntrevue
{

namespace LeetCode
{

//------------------------------------------------------------------------------
/// \url https://leetcode.com/problems/valid-parentheses/
/// \name 20. Valid Parentheses. Easy.
//------------------------------------------------------------------------------

bool is_valid_parentheses(string s)
{
  if (s.size() < 2)
  {
    return false;
  }

  stack<char> left_brackets_stack;

  //----------------------------------------------------------------------------
  /// \ref https://www.geeksforgeeks.org/unordered_map-in-cpp-stl/
  /// Implemented internally using HashTable, amoritized, search, insert, delete
  /// is O(1).
  //----------------------------------------------------------------------------

  const unordered_map<char, char> given_left_bracket {
    {'(', ')'},
    {'{', '}'},
    {'[', ']'}};
  
  const unordered_map<char, char> given_right_bracket {
    {')', '('},
    {'}', '{'},
    {']', '['}};

  // Ranged-based for loop
  for (char const& c : s)
  {
    auto is_left = given_left_bracket.find(c);
    auto is_right = given_right_bracket.find(c);

    // It's a left bracket, so add to stack.
    if (is_left != given_left_bracket.end())
    {
      left_brackets_stack.push(c);
    }
    // It's a right bracket, so it better match with the top of the stack,
    // because it's the very last left bracket and it must match with very first
    // right bracket that's approached.
    else if (is_right != given_right_bracket.end())
    {
      if (left_brackets_stack.empty())
      {
        return false;
      }

      // No match.
      if (left_brackets_stack.top() != given_right_bracket.at(c))
      {
        return false;
      }
      else
      {
        left_brackets_stack.pop();
      }
    }
    else
    {
      return false;
    }
  }

  return left_brackets_stack.empty() ? true : false;
}

//------------------------------------------------------------------------------
/// \url https://leetcode.com/problems/longest-valid-parentheses/
/// \name 32. Longest Valid Parentheses. Hard.
/// \url https://youtu.be/r0-zx5ejdq0
/// Use the fact that we keep track of indices in the stack and that we can
/// always access the value on the string through the index.
///
/// Runtime: 4 ms, faster than 95.49% of C++ online submissions for Longest
/// Valid Parentheses.
/// Memory Usage: 7.9 MB, less than 34.48% of C++ online submissions for Longest
/// Valid Parentheses.
//------------------------------------------------------------------------------
int longest_valid_parentheses(string s)
{
  if (s.size() < 2)
  {
    return 0;
  }

  // Push anytime we find brackets.
  stack<long> bracket_index_stack;

  // We have to push -1 to be the index immediately before the first index that
  // "could" be the start of valid parentheses, which is index 0.
  bracket_index_stack.push(-1);

  long longest_length {0};

  // Ranged-based for loop
  for (auto iter {s.begin()}; iter != s.end(); ++iter)
  {
    const bool is_left {*iter == '('};
    const bool is_right {*iter == ')'};
    const long string_index {distance(s.begin(), iter)};

    // It's a left bracket.
    if (is_left)
    {
      bracket_index_stack.push(string_index);
    }
    else if (is_right)
    {
      if (bracket_index_stack.top() == -1)
      {
        bracket_index_stack.push(string_index);
      }
      else
      {
        // It's a match with a left parenthesis!
        // We've used the fact that we can access the string to get its value
        // via the index that we keep in the stack.
        if (s.at(bracket_index_stack.top()) == '(')
        {
          bracket_index_stack.pop();

          // Calculate the maximum length of a valid parentheses up until now;
          // take advantage of the fact that the top of the stack will have the
          // index of the very beginning of the valid parentheses because the
          // very last index is right before the parentheses that matches.
          longest_length =
            max(longest_length, string_index - bracket_index_stack.top());
        }
        // It has to be a right parenthesis.
        else
        {
          bracket_index_stack.push(string_index);
        }
      }
    }
  }

  return longest_length;
}

//------------------------------------------------------------------------------
/// \url https://leetcode.com/problems/maximum-subarray/
/// \name 53. Maximum Subarray.
/// Assume constraint that 1 <= numslength <= 2 * 10^4
/// \details Runtime: 12 ms, faster than 74.08% of C++ online submissions for
/// Maximum Subarray. Memory Usage: 13.5 MB, less than 16.45% of C++ online
/// submissions for Maximum Subarray.
//------------------------------------------------------------------------------
int max_subarray(vector<int>& nums)
{
  if (nums.size() == 1)
  {
    return nums.at(0);
  }

  int max_sum_at_index {nums.at(0)};
  int global_max_sum {max_sum_at_index};

  for (int index {1}; index < nums.size(); ++index)
  {
    if (max_sum_at_index < 0)
    {
      max_sum_at_index = nums.at(index);
    }
    else
    {
      max_sum_at_index += nums.at(index);
    }

    // New max sum for a contiguous subarray found.
    if (max_sum_at_index > global_max_sum)
    {
      global_max_sum = max_sum_at_index;
    }
  }

  return global_max_sum;
}

//------------------------------------------------------------------------------
/// \url https://leetcode.com/problems/climbing-stairs/description/
/// \name 70. Climbing Stairs.
/// \details n = 4 case.
/// 1. 1 + 1 + 1 + 1 steps
/// 2. 2 + 1 + 1 steps
/// 3. 1 + 2 + 1 steps
/// 4. 1 + 1 + 2 steps
/// 5. 2 + 2 steps
/// 5 ways to climb to the top by considering how many ways to get to 3 steps,
/// and 2 steps, since last step to take is 1 and 2 steps, respectively.
//------------------------------------------------------------------------------
int climb_stairs(int n)
{
  return n;
}

int climb_stairs_iterative(const int n)
{
  // Assume n constrained to be 1 <= n <= 45.
  if (n < 3)
  {
    return n;
  }

  array<int, 2> steps_to_take {1, 2};

  for (int i {3}; i < n; ++i)
  {
    const int steps_to_take_to_i {steps_to_take.at(0) + steps_to_take.at(1)};
    steps_to_take.at(0) = steps_to_take.at(1);
    steps_to_take.at(1) = steps_to_take_to_i;
  }

  return steps_to_take.at(0) + steps_to_take.at(1);
}

//------------------------------------------------------------------------------
/// \url https://leetcode.com/problems/climbing-stairs/description/
/// \brief 121. Best Time to Buy and Sell Stock
//------------------------------------------------------------------------------
int max_profit(vector<int>& prices)
{
  // No profit to be had as there aren't enough days to make profit.
  if (prices.size() < 2)
  {
    return 0;
  }

  // From the definition of profit p, p := a(j) - a(i) s.t. j > i, and p >= 0,
  // so a(j) >= a(i) to qualify for profit.
  // We want to maximize profit, so we want a(j) large, constained by j > i, and
  // a(i) small, constrained by i < j.

  vector<int> min_buy_prices;

  for (auto price : prices)
  {
    if (min_buy_prices.empty())
    {
      min_buy_prices.emplace_back(price);
    }
    else
    {
      const int current_min_buy_price {min_buy_prices.back()};

      if (current_min_buy_price > price)
      {
        min_buy_prices.emplace_back(price);
      }
      else
      {
        min_buy_prices.emplace_back(current_min_buy_price);
      }
    }
  }

  //array<int, prices.size()> max_sell_prices;
  vector<int> max_sell_prices (prices.size(), 0);

  // https://stackoverflow.com/questions/3623263/reverse-iteration-with-an-unsigned-loop-variable
  for (size_t i {prices.size()}; i-- > 0; )
  {
    if (i == (prices.size() - 1))
    {
      max_sell_prices.at(i) = prices.at(i);
    }
    else
    {
      const int current_max_sell_price {max_sell_prices.at(i + 1)};

      if (current_max_sell_price < prices.at(i))
      {
        max_sell_prices.at(i) = prices.at(i);
      }
      else
      {
        max_sell_prices.at(i) = current_max_sell_price;
      }
    }
  }

  vector<int> profits;

  for (size_t i {0}; i < prices.size(); ++i)
  {
    if (profits.empty())
    {
      profits.emplace_back(0);
    }
    else
    {
      profits.emplace_back(max_sell_prices.at(i) - min_buy_prices.at(i - 1));
    }
  }

  const int maximum_profit {*max_element(profits.begin(), profits.end())};

  return maximum_profit > 0 ? maximum_profit : 0;
}

//------------------------------------------------------------------------------
/// \url https://leetcode.com/problems/coin-change/
/// \name 322. Coin Change
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
/// \details Iterative approach.
/// 1 <= coins.length <= 12
//------------------------------------------------------------------------------
int coin_change_bottom_up(std::vector<int>& coins, int amount)
{
  /*
  vector<optional<int>> min_coins_for_amount (amount + 1, nullopt);
  // Number of ways to make amount 0 with n coins is 0 (can't use coins).
  min_coins_for_amount.at(0) = 0;

  const int smallest_coinage {*min_element(coins.begin(), coins.end())};
  //const int largest_coinage {*max_element(coins.begin(), coins.end())};

  // Take amount = n. For all coin in coins, consider n - coin, n - coin is
  // amount of money left to make up the change. If n - coin > 0, recurse. If
  // n - coin == 0, done. If n - coin < 0, it's not possible.
  // Time complexity, O(amount) = O(n) for n operations.
  for (
    int amount_index {1};
    amount_index < min_coins_for_amount.size();
    ++amount_index)
  {
    // There is no way to make the amount amount_index because smallest coin is
    // too large.
    if (amount_index < smallest_coinage)
    {
      min_coins_for_amount.at(amount_index) = -1;
    }

    // Time complexity, if number of coins is m, m operations, m * n total.
    for (auto coin : coins)
    {
      if (amount_index == coin)
      {
        min_coins_for_amount.at(amount_index);
      }
    }
  }
  */

  // \url https://leetcode.com/problems/coin-change/discuss/77360/C%2B%2B-O(n*amount)-time-O(amount)-space-DP-solution
  // O(n*amount) time O(amount) space Dynamic Programming solution.

  vector<int> min_coins_for_amount (amount + 1, amount + 1);
  // Set 0 amount to be made up of 0 coins.
  min_coins_for_amount.at(0) = 0;

  for (int amount_index {1}; amount_index < amount + 1; ++amount_index)
  {
    for (auto coin : coins)
    {
      if (coin == amount_index)
      {
        // Only 1 way to make this amount with smallest amount of coin, no less
        // number of coins than 1.
        min_coins_for_amount.at(amount_index) = 1;
        break;
      }
      else if (coin < amount_index)
      {
        // Compare the value stored against this value made with this coin. 

        min_coins_for_amount.at(amount_index) = min(
          min_coins_for_amount.at(amount_index),
          min_coins_for_amount.at(amount_index - coin) + 1);
      }
    }
  }

  return min_coins_for_amount.at(amount) < amount + 1 ?
    min_coins_for_amount.at(amount) : -1;
}

int coin_change_top_down(vector<int>& coins, int amount)
{
  map<int, int> min_coins;
  min_coins[0] = 0;

  return coin_change_top_down_step(coins, amount, min_coins);
}

// \url https://backtobackswe.com/platform/content/the-change-making-problem/solutions

int coin_change_top_down_step(
  vector<int>& coins,
  int amount,
  map<int, int>& min_coins)
{
  // Negative value base case, necessary to deal with amount < coin.
  if (amount < 0)
  {
    return -1;
  }

  if (amount < 1)
  {
    return 0;
  }

  // Found amount already in map.
  if (min_coins.find(amount) != min_coins.end())
  {
    return min_coins.at(amount);
  }

  int minimum_to_amount {amount + 1};

  for (int coin : coins)
  {
    /*
    if (amount > coin)
    {
      if (min_coins.find(amount) != min_coins.end())
      {
        min_coins[amount] =
          min(min_coins[amount],
            // Add 1 to account for the step or coin that needs to be added to
            // make amount.
            coin_change_top_down_step(
              coins,
              (amount - coin),
              min_coins) + 1);
      }
      else
      {
        min_coins[amount] =
          coin_change_top_down_step(coins, (amount - coin), min_coins) + 1;
      }
    }
    else if (amount == coin)
    {
      min_coins[amount] = 1;
    }
    */
    const int previous_coin_number {
      coin_change_top_down_step(coins, amount - coin, min_coins)};

    if ((previous_coin_number >= 0) &&
      (minimum_to_amount > (previous_coin_number + 1)))
    {
      minimum_to_amount = previous_coin_number + 1;
    }

    // Otherwise, if previous_coin_number < 0, then there was no way to form a
    // coin set to make amount - coin.
  }

  if (minimum_to_amount == amount + 1)
  {
    minimum_to_amount = -1;
  }

  min_coins[amount] = minimum_to_amount;

  // True if we've found a minimum number for amount.
  //return (min_coins.find(amount) != min_coins.end()) ?
    //min_coins.at(amount) : -1;
  return minimum_to_amount;
}

//------------------------------------------------------------------------------
/// \details Recursive approach.
/// Too slow.
//------------------------------------------------------------------------------
int min_coin_change_recursive_step(
  vector<int>& coins,
  const int smallest_coinage,
  vector<optional<int>>& min_coins_for_amount,
  int amount)
{
  if (min_coins_for_amount.at(amount).has_value())
  {
    return *min_coins_for_amount.at(amount);
  }

  if (amount < smallest_coinage)
  {
    min_coins_for_amount.at(amount) = -1;
    return -1;
  }

  // amount was found amonst the coinage; so 1 coin can make the amount and
  // we're done; no other way to make the amount with less number of coins.
  if (find(coins.begin(), coins.end(), amount) != coins.end())
  {
    min_coins_for_amount.at(amount) = 1;
    return 1;
  }

  vector<int> minimum_coins_change;

  for (auto coin : coins)
  {
    if (amount > coin)
    {
      // amount - coin represents the remaining amount of money to make change.
      const int remaining {amount - coin};

      const int minimum_number_of_coins {
        min_coin_change_recursive_step(
          coins,
          smallest_coinage,
          min_coins_for_amount,
          remaining)};

      minimum_coins_change.emplace_back(minimum_number_of_coins);
    }
  }

  if (minimum_coins_change.empty())
  {
    min_coins_for_amount.at(amount) = -1;
    return -1;
  }

  int minimum_value {amount + 1};
  int second_minimum_value {amount + 2};

  for (auto n : minimum_coins_change)
  {
    if (n < minimum_value)
    {
      minimum_value = n;
    }

    if ((minimum_value <  n) && (n < second_minimum_value))
    {
      second_minimum_value = n;
    }
  }

  if (minimum_value == -1)
  {
    if (second_minimum_value < (amount + 2))
    {
      // Remember to add 1 coin for the coin we had to have considered adding to
      // get to the given amount.

      min_coins_for_amount.at(amount) = second_minimum_value + 1;
      return second_minimum_value + 1;
    }
    else
    {
      min_coins_for_amount.at(amount) == -1;
      return -1;
    }
  }
  else
  {
    // Remember to add 1 coin for the coin we had to have considered adding to 
    // get to the amount amount.
    min_coins_for_amount.at(amount) = minimum_value + 1;
    return minimum_value + 1;
  }
}

int coin_change_recursive(std::vector<int>& coins, int amount)
{
  vector<optional<int>> min_coins_for_amount (amount + 1, nullopt);
  min_coins_for_amount.at(0) = 0;

  const int smallest_coinage {*min_element(coins.begin(), coins.end())};

  return
    min_coin_change_recursive_step(
      coins,
      smallest_coinage,
      min_coins_for_amount,
      amount);
}

//------------------------------------------------------------------------------
/// \url https://leetcode.com/problems/max-sum-of-rectangle-no-larger-than-k/
/// \name 363. Max Sum of Rectangle No Larger Than K.
/// \ref https://www.youtube.com/watch?v=-FgseNO-6Gk
/// Back To Back SWE, Maximum Sum Rectangle In A 2D Matrix - Kadane's Algorithm
//------------------------------------------------------------------------------
int max_sum_submatrix(vector<vector<int>>& matrix)
{
  // (Max) size in row, M.
  const size_t M {matrix.size()};
  const size_t N {matrix.at(0).size()};

  if (M == 1 && N == 1)
  {
    return matrix.at(0).at(0);
  }

  //int max_sum_per_row {matrix.at(0).at(0)}; 

  // Includes the max sum for row.
  //int max_sum_per_column {max_sum_per_row};
  int global_max_sum {matrix.at(0).at(0)};

  size_t J_column_left {0};
  // J_column_left <= J_column_right always.
  size_t J_column_right {J_column_left};
  size_t I_row_top {0};
  // I_row_top <= I_row_bottom always.
  size_t I_row_bottom {I_row_bottom};

  // Running sum of each column; running column sums, i.e. for each j.
  vector<int> running_column_sum (N, 0);

  // Iterate over a possible whole matrix from I_row_top to I_row_bottom, with
  // I_row_top <= I_row_bottom.

  for (size_t i_row_top {0}; i_row_top < M; ++i_row_top)
  {
    for (size_t i_row_bottom {i_row_top}; i_row_bottom < M; ++i_row_bottom)
    {
      if (i_row_bottom == i_row_top)
      {  
        running_column_sum = matrix.at(i_row_top);
      }
      else
      {
        // Add the new row to the running column sum.
        
        for (int j {0}; j < N; ++j)
        {
          running_column_sum.at(j) += matrix.at(i_row_bottom).at(j);
        }
        //transform(
        //  running_column_sum.begin(),
        //  running_column_sum.end(),
        //  matrix.at(i_row_bottom).begin(),
        //  running_column_sum.end(),
        //  plus<int>());
      }

      const auto result = find_subrow_max(running_column_sum);

      if (result.first > global_max_sum)
      {
        global_max_sum = result.first;

        I_row_top = i_row_top;
        I_row_bottom = i_row_bottom;
        J_column_left = result.second.first;
        J_column_right = result.second.second;
      }
    }
  }

  return global_max_sum;
}

// Kadane's algorithm.
pair<int, pair<size_t, size_t>> find_subrow_max(vector<int>& row)
{
  const size_t N {row.size()};

  if (N == 1)
  {
    return make_pair<int, pair<size_t, size_t>>(
      move(row.at(0)),
      make_pair<size_t, size_t>(0, 0));
  }

  int max_sum_per_column {row.at(0)};
  int global_row_sum {max_sum_per_column};

  size_t global_start_index {0};
  size_t global_end_index {global_start_index};

  size_t start_index {0};
  size_t end_index {start_index};

  for (size_t j {1}; j < N; ++j)
  {
    if (max_sum_per_column < 0)
    {
      max_sum_per_column = row.at(j);

      start_index = j;
      end_index = j;
    }
    else
    {
      max_sum_per_column += row.at(j);
      end_index = j;
    }

    if (global_row_sum < max_sum_per_column)
    {
      global_row_sum = max_sum_per_column;

      global_start_index = start_index;
      global_end_index = end_index;
    }
  }

  return make_pair<int, pair<size_t, size_t>>(
    move(global_row_sum),
    make_pair<size_t, size_t>(
      move(global_start_index),
      move(global_end_index)));
}


//------------------------------------------------------------------------------
/// \url https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/
/// \name 647. Palindromic Substrings.
/// \brief Given a string, your task is to count how many palindromic substrings
/// in this string.
//------------------------------------------------------------------------------
int count_palindromic_substrings(string s)
{
  if (s.size() < 2)
  {
    return s.size();
  }

  int count {0};

  auto start_iter = s.begin();

  while (start_iter != s.end())
  {
    count += find_even_size_palindromes(s.begin(), s.end(), start_iter);

    count += find_odd_size_palindromes(s.begin(), s.end(), start_iter);

    ++start_iter;
  }

  return count;
}

int count_palindromic_substrings_simple(string s)
{
  int count {0};
  // Let N be the length of the string.
  const long N {distance(s.begin(), s.end())};

  // The middle of the palindrome could be in 1 of 2N-1 total positions; either
  // at the letter, which will be when "center" is even, or in between 2
  // letters, which is N - 1 positions (when center is odd).
  for (long center {0}; center < (2 * N - 1); ++center)
  {
    long left {center / 2};
    // If center is even, then it's an odd-length palindrome, possibly.
    // If center is odd, then it's an even-length palindrome, possibly.
    long right {left + center % 2};

    while (left >= 0 && right < N)
    {
      if (s.at(left) == s.at(right))
      {
        // Palindrome found.
        ++count;

        // Continue, since we did not not find a palindrome.
        --left;
        ++right;
      }
      else
      {
        break;
      }
    }
  }

  return count;
}

} // namespace LeetCode
} // namespace QuestionsDEntrevue
