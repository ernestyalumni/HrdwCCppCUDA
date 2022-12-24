/// \url https://stackoverflow.com/questions/28136739/is-it-possible-to-control-the-size-of-an-array-using-the-type-parameter-of-a-gen
/// cf. https://doc.rust-lang.org/std/marker/trait.Sized.html
/// Trait std::marker::Sized
/// pub trait Sized { }
/// Types with a constant size known at compile time.
pub struct FixedSizeArray<T: Sized, const N: usize>
{
  pub data_: [T; N],
}

#[cfg(test)]
mod tests
{
  // The tests module is a regular module that follows the usual visibility rules in Ch. 7,
  // "Paths for Referring to an Item in the Module Tree" of Rust book.
  // Because tests module is an inner module, we need to bring the code under test in the outer
  // module into scope of inner module.
  use super::*;

  #[test]
  fn fixed_size_array_constructs()
  {
    {
      let a = FixedSizeArray::<i32, 3> {data_: [1, 2, 3]};
      assert_eq!(a.data_[0], 1);
      assert_eq!(a.data_[1], 2);
      assert_eq!(a.data_[2], 3);
    }
  }

}