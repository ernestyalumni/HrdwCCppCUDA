pub struct ThreeSum;

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