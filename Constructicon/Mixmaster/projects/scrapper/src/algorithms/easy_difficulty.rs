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
}