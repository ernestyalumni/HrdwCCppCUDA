use std::collections::HashMap;

/// 3. Longest Substring Without Repeating Characters
pub struct LengthOfLongestSubstring;
pub struct ThreeSum;
/// 11. Container With Most Water
pub struct ContainerWithMostWater;

//------------------------------------------------------------------------------
/// 3. Longest Substring Without Repeating Characters
/// https://leetcode.com/problems/longest-substring-without-repeating-characters/
/// Given a string s, find the length of the longest without duplicate
/// characters.
//------------------------------------------------------------------------------
impl LengthOfLongestSubstring
{
  pub fn length_of_longest_substring(s: String) -> i32
  {
    // Map each character seen before to its index along the string.
    let mut seen_character_to_index = HashMap::new();

    let mut max_length = 0;
    // The start index of the substring under current consideration.
    let mut start_index = 0;

    for (i, c) in s.chars().enumerate()
    {
      // If we've already seen the character before, then we need to start a 
      // new substring. Do this by updating the start_index.
      if let Some(&last_index) = seen_character_to_index.get(&c)
      {
        // Move the start of the new substring to consider to the immediate
        // right of the last occurence of the seen, repeated character.
        start_index = start_index.max(last_index + 1);
      }

      seen_character_to_index.insert(c, i);

      max_length = max_length.max(i - start_index + 1);
    }

    max_length.try_into().unwrap()
  }
}

//------------------------------------------------------------------------------
/// 11. Container With Most Water
/// You are given an integer array height of length n. There are n vertical
/// lines drawn such that the two endpoints of the ith line are (i, 0) and 
/// (i, height[i]).
///
/// Find two lines that together with the x-axis form a container, such that the
/// container contains the most water.
///
/// Return the maximum amount of water a container can store.
///
/// Notice that you may not slant the container.
///
/// Constraints:
///
/// * n == height.length
/// * 2 <= n <= 105
/// * 0 <= height[i] <= 104
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
/// 15. 3Sum.
/// Given an integer array nums, return all triplets.
/// Key insight: sort first. Then iterate on each element.
//------------------------------------------------------------------------------
impl ThreeSum
{
  pub fn three_sum(nums: Vec<i32>) -> Vec<Vec<i32>>
	{
    let mut nums = nums.clone();

    // O(N log N) time complexity.
    nums.sort_unstable();

    let mut triplets = Vec::new();

    let n = nums.len();

    for i in 0..n - 2
    {
      // Because we've sorted nums, we know that any nums[j] where j > i will be
      // positive-valued and couldn't add up to 0 (need a negative number).
      if nums[i] > 0
      {
        break;
      }

      // Skip duplicates.
      if i > 0 && nums[i] == nums[i - 1]
      {
        continue;
      }

      let mut l = i + 1;
      let mut r = n - 1;

      while l < r
      {
        let two_sum = nums[l] + nums[r];
        if two_sum + nums[i] == 0
        {
          triplets.push(vec![nums[i], nums[l], nums[r]]);

          // Skip over duplicates on the left and right.
          while l < r && nums[l] == nums[l + 1]
          {
            l += 1;
          }
          while l < r && nums[r] == nums[r - 1]
          {
            r -= 1;
          }

          l += 1;
          r -= 1;
        }
        else if two_sum + nums[i] < 0
        {
          l += 1;
        }
        // two_sum + nums[i] > 0
        else
        {
          r -= 1;
        }
      }
    }

    triplets
  }
}