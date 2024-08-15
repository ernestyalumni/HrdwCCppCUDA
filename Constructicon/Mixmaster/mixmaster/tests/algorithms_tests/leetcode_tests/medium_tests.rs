#[cfg(test)]
pub mod medium_tests
{
  use mixmaster::algorithms::leetcode::medium::{
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