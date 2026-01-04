#[cfg(test)]
pub mod medium_tests
{
  use mixmaster::algorithms::leetcode::medium::{
    // 3. Longest Substring Without Repeating Characters
    LengthOfLongestSubstring,
    // 11. Container With Most Water
    ContainerWithMostWater,
    // 15.
    ThreeSum,
  };

  use std::collections::BTreeSet;

  fn to_btree_set(vecs: Vec<Vec<i32>>) -> BTreeSet<BTreeSet<i32>>
  {
    vecs.into_iter()
      .map(|vec| vec.into_iter().collect::<BTreeSet<_>>())
      .collect::<BTreeSet<_>>()
  }

  //----------------------------------------------------------------------------
  /// 3. Longest Substring Without Repeating Characters
  /// https://leetcode.com/problems/longest-substring-without-repeating-characters/
  //----------------------------------------------------------------------------
  #[test]
  fn test_length_of_longest_substring()
  {
    // Example 1
    let s = "abcabcbb".to_string();
    let expected = 3;
    let result = LengthOfLongestSubstring::length_of_longest_substring(s);
    assert_eq!(result, expected);

    // Example 2
    let s = "bbbbb".to_string();
    let expected = 1;
    let result = LengthOfLongestSubstring::length_of_longest_substring(s);
    assert_eq!(result, expected);

    // Example 3
    let s = "pwwkew".to_string();
    let expected = 3;
    let result = LengthOfLongestSubstring::length_of_longest_substring(s);
    assert_eq!(result, expected);

    // Example 4
    let s = "tmmzuxt".to_string();
    let expected = 5;
    let result = LengthOfLongestSubstring::length_of_longest_substring(s);
    assert_eq!(result, expected);
  }

  //----------------------------------------------------------------------------
  /// https://leetcode.com/problems/container-with-most-water/description/
  /// 11. Container With Most Water
  /// n == height.length
  /// * 2 <= n <= 105
  /// * 0 <= height[i] <= 104
  //----------------------------------------------------------------------------
  #[test]
  fn test_container_with_most_water()
  {
    // Example 1
    // Input: height = [1,8,6,2,5,4,8,3,7]
    // Output: 49
    let height = vec![1,8,6,2,5,4,8,3,7];
    let expected = 49;
    let result = ContainerWithMostWater::max_area(height);
    assert_eq!(result, expected);

    // Example 2:
    // Input: height = [1,1]
    // Output: 1
    let height = vec![1,1];
    let expected = 1;
    let result = ContainerWithMostWater::max_area(height);
    assert_eq!(result, expected);
  }

  //----------------------------------------------------------------------------
  /// 15. 3Sum
  //----------------------------------------------------------------------------
  #[test]
  fn test_three_sum()
  {
    // Example 1
    let nums = vec![-1, 0, 1, 2, -1, -4];
    let expected = vec![vec![-1, -1, 2], vec![-1, 0, 1]];

    let result = ThreeSum::three_sum(nums);

    assert_eq!(to_btree_set(result), to_btree_set(expected));

    // Example 2
    let nums = vec![0,1,1];
    let expected = vec![];
    let result = ThreeSum::three_sum(nums);
    assert_eq!(to_btree_set(result), to_btree_set(expected));

    // Example 3
    let nums = vec![0,0,0];
    let expected = vec![vec![0,0,0]];
    let result = ThreeSum::three_sum(nums);
    assert_eq!(to_btree_set(result), to_btree_set(expected));
  }
}
