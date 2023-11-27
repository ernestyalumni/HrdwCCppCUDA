//------------------------------------------------------------------------------
/// \details & denotes a reference. It's more efficient to pass a pointer to a
/// slice of the data. The function can read or "borrow" from slice without
/// taking ownership.
//------------------------------------------------------------------------------
fn validate_subsequence(array: &[i32], sequence: &[i32]) -> bool
{
  let mut array_index = 0;
  let mut sequence_index = 0;

  while array_index < array.len() && sequence_index < sequence.len()
  {
    if array[array_index] == sequence[sequence_index]
    {
      sequence_index += 1;
    }
    array_index += 1;
  }

  sequence_index == sequence.len()
}

fn transpose_matrix(matrix: Vec<Vec<i32>>) -> Vec<Vec<i32>>
{
  let m = matrix.len();
  let n = matrix[0].len();

  // 
  let mut transposed_matrix = vec![vec![0; m]; n];

  for i in 0..n
  {
    for j in 0..m
    {
      transposed_matrix[i][j] = matrix[j][i];
    }
  }

  transposed_matrix
}

#[cfg(test)]
mod tests
{
  use super::*;

  #[test]
  fn test_validate_subsequence()
  {
    assert_eq!(
      validate_subsequence(&[5, 1, 22, 25, 6, -1, 8, 10], &[1, 6, -1, 10]),
      true);
    assert_eq!(
      validate_subsequence(&[5, 1, 22, 25, 6, -1, 8, 10], &[1, 6, 10]),
      true);
    assert_eq!(
      validate_subsequence(&[5, 1, 22, 25, 6, -1, 8, 10], &[1, 6, 11]),
      false);
    assert_eq!(validate_subsequence(&[1, 2, 3, 4], &[1, 3, 4]), true);
    assert_eq!(validate_subsequence(&[1, 2, 3, 4], &[2, 4]), true);
    assert_eq!(validate_subsequence(&[1, 2, 3, 4], &[1, 3, 5]), false);
  }

  #[test]
  fn test_transpose_rectangular_matrix()
  {
    let matrix = vec![vec![1, 2, 3], vec![4, 5, 6]];
    let expected = vec![vec![1, 4], vec![2, 5], vec![3, 6]];

    assert_eq!(transpose_matrix(matrix), expected);
  }
}