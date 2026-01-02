#[cfg(test)]
pub mod easy_tests
{
  use mixmaster::algorithms::leetcode::easy::{
    // 1.
    TwoSum,
    // 20.
    ValidParentheses,
    // 121.
    BestTimeToBuyAndSellStock,
    // 125.
    ValidPalindrome,
    // 217.
    ContainsDuplicate,
  };

  use std::collections::HashSet;

  //----------------------------------------------------------------------------
  /// 1. Two Sum
  //----------------------------------------------------------------------------
  #[test]
  fn test_two_sum_brute_force()
  {
    // Example 1
    let nums = vec![2, 7, 11, 15];
    let target = 9;
    let expected: HashSet<usize> = [0, 1].iter().cloned().collect();

    let result = TwoSum::brute_force(nums, target);
    let result_set: HashSet<usize> = result.into_iter().collect();

    assert_eq!(result_set, expected);

    // Example 2
    let nums = vec![3, 2, 4];
    let target = 6;
    let expected: HashSet<usize> = [1, 2].iter().cloned().collect();

    let result = TwoSum::brute_force(nums, target);
    let result_set: HashSet<usize> = result.into_iter().collect();

    assert_eq!(result_set, expected);

    // Example 3
    let nums = vec![3, 3];
    let target = 6;
    let expected: HashSet<usize> = [0, 1].iter().cloned().collect();

    let result = TwoSum::brute_force(nums, target);
    let result_set: HashSet<usize> = result.into_iter().collect();

    assert_eq!(result_set, expected);
  }

  #[test]
  fn test_two_sum_with_map()
  {
    // Example 1
    let nums = vec![2, 7, 11, 15];
    let target = 9;
    let expected: HashSet<usize> = [0, 1].iter().cloned().collect();

    let result = TwoSum::two_sum_with_map(nums, target);
    let result_set: HashSet<usize> = result.into_iter().collect();

    assert_eq!(result_set, expected);

    // Example 2
    let nums = vec![3, 2, 4];
    let target = 6;
    let expected: HashSet<usize> = [1, 2].iter().cloned().collect();

    let result = TwoSum::two_sum_with_map(nums, target);
    let result_set: HashSet<usize> = result.into_iter().collect();

    assert_eq!(result_set, expected);

    // Example 3
    let nums = vec![3, 3];
    let target = 6;
    let expected: HashSet<usize> = [0, 1].iter().cloned().collect();

    let result = TwoSum::two_sum_with_map(nums, target);
    let result_set: HashSet<usize> = result.into_iter().collect();

    assert_eq!(result_set, expected);
  }

  //----------------------------------------------------------------------------
  /// 20. Valid Parentheses
  //----------------------------------------------------------------------------
  #[test]
  fn test_is_valid_parentheses()
  {
    // Example 1
    let s = String::from("()");
    let expected = true;
    assert_eq!(ValidParentheses::is_valid_parentheses(s), expected);

    // Example 2
    let s = String::from("()[]{}");
    let expected = true;
    assert_eq!(ValidParentheses::is_valid_parentheses(s), expected);

    // Example 3
    let s = String::from("(]");
    let expected = false;
    assert_eq!(ValidParentheses::is_valid_parentheses(s), expected);
  }

  //----------------------------------------------------------------------------
  /// 121. Best Time to Buy and Sell Stock
  //----------------------------------------------------------------------------
  #[test]
  fn test_max_profit()
  {
    // Example 1
    let prices = vec![7,1,5,3,6,4];
    let expected = 5;
    assert_eq!(BestTimeToBuyAndSellStock::max_profit(prices), expected);

    // Example 2
    let prices = vec![7,6,4,3,1];
    let expected = 0;
    assert_eq!(BestTimeToBuyAndSellStock::max_profit(prices), expected);
  }

  //----------------------------------------------------------------------------
  /// 125. Valid Palindrome
  //----------------------------------------------------------------------------
  #[test]
  fn test_is_palindrome()
  {
    let input = String::from("A man, a plan, a canal: Panama");

    assert!(ValidPalindrome::is_palindrome(input));

    let input = String::from("race a car");

    assert!(!ValidPalindrome::is_palindrome(input));

    let input = String::from(" ");

    assert!(ValidPalindrome::is_palindrome(input));

    // https://neetcode.io/problems/is-palindrome

    let input = String::from("Was it a car or a cat I saw?");

    assert!(ValidPalindrome::is_palindrome(input));

    let input = String::from("tab a cat");

    assert!(!ValidPalindrome::is_palindrome(input));
  }

  //----------------------------------------------------------------------------
  /// https://leetcode.com/problems/contains-duplicate/
  /// 217. Contains Duplicate
  //----------------------------------------------------------------------------
  #[test]
  fn test_contains_duplicate_with_hashset()
  {
    // Example 1
    let nums = vec![1,2,3,1];
    assert!(ContainsDuplicate::contains_duplicate_with_hashset(nums));

    // Example 2
    let nums = vec![1,2,3,4];
    assert!(!ContainsDuplicate::contains_duplicate_with_hashset(nums));

    // Example 3
    let nums = vec![1,1,1,3,3,4,3,2,4,2];
    assert!(ContainsDuplicate::contains_duplicate_with_hashset(nums));
  }
}

