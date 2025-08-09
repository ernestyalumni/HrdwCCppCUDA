pub struct TwoSum;
pub struct ValidParentheses;
pub struct BestTimeToBuyAndSellStock;
// 125. Valid Palindrome
pub struct ValidPalindrome;

//------------------------------------------------------------------------------
/// 1. Two Sum.
/// Assume each input has exactly one solution, you may not use the same element
/// twice.
//------------------------------------------------------------------------------
impl TwoSum
{
  pub fn brute_force(nums: Vec<i32>, target: i32) -> Vec<usize>
  {
    let n = nums.len();

    // O(N^2) time complexity.
    for i in 0..n-1
    {
      for j in i + 1..n
      {
        if nums[i] + nums[j] == target
        {
          return vec![i, j];
        }
      }
    }

    vec![]
  }

  pub fn two_sum_with_map(nums: Vec<i32>, target_sum: i32) -> Vec<usize>
  {
    use std::collections::HashMap;

    let mut value_and_indicies = HashMap::new();

    for (i, &num) in nums.iter().enumerate()
    {
      let complement = target_sum - num;

      if let Some(&complement_index) = value_and_indicies.get(&complement)
      {
        // ! indicates this is a macro, vec! is convenient for creating a vector
        // with known values.
        return vec![i, complement_index];
      }
      else
      {
        value_and_indicies.insert(num, i);
      }
    }

    vec![]
  }
}

impl ValidParentheses
{
  pub fn is_valid_parentheses(s: String) -> bool
  {
    use std::collections::HashMap;

    // Early return optimization, check whether string has odd length upfront.
    if s.len() % 2 != 0
    {
      return false;
    }

    // Rust Vec has pop and push.
    let mut stack: Vec<char> = Vec::new();

    // Just use the stack for closing brackets.
    let bracket_map: HashMap<char, char> = [
      ('(', ')'),
      ('{', '}'),
      ('[', ']'),
    ].iter().cloned().collect();

    for c in s.chars()
    {
      if let Some(&closing) = bracket_map.get(&c)
      {
        stack.push(closing);
      }
      else
      {
        if stack.pop() != Some(c)
        {
          return false;
        }
      }
    }

    stack.is_empty()
  }
}

//------------------------------------------------------------------------------
/// 121. Best Time to Buy and Sell Stock.
/// You want to maximize your profit by choosing a single day to buy one stock
/// and choosing a different day in the future to sell that stock.
///
/// Key idea: at each step, update profit for maximum profit and minimum price
/// in that order.
//------------------------------------------------------------------------------
impl BestTimeToBuyAndSellStock
{
  pub fn max_profit(prices: Vec<i32>) -> i32
  {
    let mut minimum_price = prices[0];
    let mut profit = 0;

    for current_price in prices
    {
      let current_profit = current_price - minimum_price;

      if current_profit > profit
      {
        profit = current_profit;
      }

      if minimum_price > current_price
      {
        minimum_price = current_price;
      }
    }

    profit
  }
}

//------------------------------------------------------------------------------
/// 125. Valid Palindrome
/// A phrase is a palindrome if, after converting all uppercase letters into
/// lowercase letters and removing all non-alphanumeric characters, it reads the
/// same forward and backward. Alphanumeric characters include letters and
/// numbers.
/// s consists only of printable ASCII characters.
/// https://leetcode.com/problems/valid-palindrome/description/
//------------------------------------------------------------------------------
impl ValidPalindrome
{
  //----------------------------------------------------------------------------
  /// 
  //----------------------------------------------------------------------------
  pub fn is_palindrome(s: String) -> bool
  {
    let N = s.len();
    let mut l_ptr = 0;
    let mut r_ptr = N - 1;

    // String is a collection of UTF-8 encoded bytes, and not single char
    // elements.
    let chars: Vec<char> = s.chars().collect();

    let mut result = true;

    while l_ptr < r_ptr
    {
      // Skip non-alphanumeric characters on the left.
      while l_ptr < r_ptr && !chars[l_ptr].is_alphanumeric()
      {
        l_ptr += 1;
      }

      // Skip non-alphanumeric characters on the right.
      while r_ptr > l_ptr && !chars[r_ptr].is_alphanumeric()
      {
        r_ptr -= 1;
      }

      // If pointers have crossed, break out of loop.
      if l_ptr >= r_ptr
      {
        break;
      }

      if chars[l_ptr].to_ascii_lowercase() !=
        chars[r_ptr].to_ascii_lowercase()
      {
          result = false;
          break;
      }

      l_ptr += 1;
      r_ptr -= 1;
    }

    result
  }
}
