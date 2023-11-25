#![feature(alloc)]

#[cfg(test)]
mod tests
{
  //-----------------------------------------------------------------------------------------------
  /// cf. https://docs.rust-embedded.org/book/collections/
  /// Import alloc crate directly,
  //-----------------------------------------------------------------------------------------------
  extern crate alloc;

  // cf. https://doc.rust-lang.org/alloc/vec/struct.Vec.html
  #[test]
  fn alloc_vec_constructs()
  {
    // cf. https://doc.rust-lang.org/alloc/vec/index.html
    let mut v = alloc::vec::Vec::new();

    assert!(v.is_empty());
    v.push(3);

    assert!(!v.is_empty());

    assert_eq!(v[0], 3);
  }
}