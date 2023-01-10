//-------------------------------------------------------------------------------------------------
/// cf. https://doc.rust-lang.org/rust-by-example/custom_types/enum/testcase_linked_list.html
//-------------------------------------------------------------------------------------------------

use crate::data_structures::lists::enum_as_linked_list::List::{Cons, Nil};

enum List<T>
{
  // Cons: Tuple struct that wraps an element and a pointer to the next node
  Cons(T, Box<List<T>>),
  // Nil: A node that signifies the end of the linked list
  Nil,
}

// Methods can be attached to an enum.
impl<T> List<T>
{
  // Create an empty list.
  fn new() -> List<T>
  {
    // `Nil` has type `List`
    Nil
  }

  // Consume a list, and return the same list with a new element at its front.
  fn prepend(self, elem: T) -> List<T>
  {
    // `Cons` also has type List
    Cons(elem, Box::new(self))
  }

  // Return the length of the list.
  fn len(&self) -> usize
  {
    // `self` has to be matched, because the behavior of this method depends on the variant of
    // `self`.
    // `self` has type `&List`, and `*self` has type `List`, matching on a concrete type `T` is
    // preferred over a match on a reference `&T` after Rust 2018 you can use self here and tail
    // (with no ref) below as well, rust will infer &s and ref tail.
    // See https://doc.rust-lang.org/edition-guide/rust-2018/ownership-and-lifetimes/default-match-bindings.html
    match *self
    {
      // Can't take ownership of the tail, because `self` is borrowed; isntead take a reference to
      // the tail.
      Cons(_, ref tail) => 1 + tail.len(),
      // Base Case: An empty list has zero length
      Nil => 0, 
    }
  }
}

#[cfg(test)]
mod tests
{
  use super::*;

  #[test]
  fn empty_list_constructs_with_new()
  {
    let x: List::<u32> = List::<u32>::new();
    assert!(matches!(x, Nil));
  }

  #[test]
  fn list_with_prepended_values_constructs()
  {
    // Create an empty linked list.
    let mut x = List::<u32>::new();

    // Prepend some elements.
    x = x.prepend(1);
    x = x.prepend(2);
    x = x.prepend(3);

    // Show the final state of the list.

    assert_eq!(x.len(), 3);

    // cf. https://stackoverflow.com/questions/9109872/how-do-you-access-enum-values-in-rust
    // We're only interested in matching one of the variants, use if let.
    if let List::<u32>::Cons(value, x1) = x
    {
      assert_eq!(value, 3);

      if let List::<u32>::Cons(value, x2) = *x1
      {
        assert_eq!(value, 2);

        if let List::<u32>::Cons(value, x3) = *x2
        {
          assert_eq!(value, 1);

          assert!(matches!(*x3, Nil));
        }
      }
    }
  }
}