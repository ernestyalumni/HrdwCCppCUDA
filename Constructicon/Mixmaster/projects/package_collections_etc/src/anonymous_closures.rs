use gag::BufferRedirect;
use std::io::Read;

#[derive(Debug, PartialEq, Copy, Clone)]
enum ShirtColor
{
  Red,
  Blue,
}

struct Inventory
{
  shirts: Vec<ShirtColor>,
}

impl Inventory
{
  fn giveaway(&self, user_preference: Option<ShirtColor>) -> ShirtColor
  {
    // The unwrap_or_else method on Option<T> takes 1 argument: a *closure* without any arguments
    // that returns a value T (same type stored in the Some variant of the Option<T>)
    //
    // We specify the closure expression || self.most_stocked() as argument to unwrap_or_else.
    // implementation of unwrap_or_else will evaluate the closure later if result is needed.
    // On interesting aspect is that we've passed a closure that calls self.most_stocked() on the
    // current Inventory instance.
    // The closure captures an immutable reference to the self Inventory instance and passes it
    // with the code we specify to the unwrap_or_else method. Functions, on the other hand, aren't
    // able to capture their environment in this way.
    user_preference.unwrap_or_else(|| self.most_stocked())
  }

  fn most_stocked(&self) -> ShirtColor
  {
    let mut num_red = 0;
    let mut num_blue = 0;

    for color in &self.shirts
    {
      match color
      {
        ShirtColor::Red => num_red += 1,
        ShirtColor::Blue => num_blue +=1,
      }
    }
    if num_red > num_blue
    {
      ShirtColor::Red
    }
    else
    {
      ShirtColor::Blue
    }
  }
}

//-------------------------------------------------------------------------------------------------
/// \url https://doc.rust-lang.org/book/ch13-01-closures.html
/// \details sort_by_key uses FnMut instead of FnOnce for trait bound.
//-------------------------------------------------------------------------------------------------

struct Rectangle
{
  width: u32,
  height: u32,
}

#[cfg(test)]
mod tests
{
  use super::*;

  // cf. https://stackoverflow.com/questions/46378637/how-to-make-a-variable-with-a-scope-lifecycle-for-all-test-functions-in-a-rust-t

  struct Setup
  {
    store: Inventory,
  }

  impl Setup
  {
    fn new() -> Self
    {
      Self
      {
        store: Inventory {
          shirts: vec![ShirtColor::Blue, ShirtColor::Red, ShirtColor::Blue],
        }
      }
    }
  }

  #[test]
  fn user_with_preference_gets_preference()
  {
    let setup = Setup::new();

    // cf. https://doc.rust-lang.org/std/option/
    // Type Option represents an optional value: every Option is either Some and contains a value,
    // or None, and doesn't.
    let user_pref1 = Some(ShirtColor::Red);
    let giveaway1 = setup.store.giveaway(user_pref1);

    assert_eq!(giveaway1, ShirtColor::Red);
  }

  #[test]
  fn user_with_no_preference_gets_most_stocked()
  {
    let setup = Setup::new();

    let user_pref2 = None;
    let giveaway2 = setup.store.giveaway(user_pref2);

    assert_eq!(giveaway2, ShirtColor::Blue);
  }

  //-----------------------------------------------------------------------------------------------
  /// \url https://doc.rust-lang.org/book/ch13-01-closures.html#capturing-references-or-moving-ownership
  /// \brief Capturing References or Moving Ownership
  //-----------------------------------------------------------------------------------------------

  // Works only if this is run: $ cargo test -- --nocapture
  // since unit tests in rust cargo suppresses stdout, i.e. println!.
  // Uncomment this out to run it.
  /*
  #[test]
  fn closure_can_capture_immutable_reference()
  {

    // https://users.rust-lang.org/t/how-to-test-functions-that-use-println/67188/3
    // https://doc.rust-lang.org/nightly/std/result/enum.Result.html#method.unwrap
    // Returns the contained Ok value, consuming the self value.
    let mut buffer = BufferRedirect::stdout().unwrap();

    let list = vec![1, 2, 3];
    // Before defining closure:
    assert_eq!(list, vec![1, 2, 3]);

    let only_borrows = || println!("From closure: {:?}", list);

    // Before calling closure:
    assert_eq!(list, vec![1, 2, 3]);

    only_borrows();    

    let mut output = String::new();
    buffer.read_to_string(&mut output).unwrap();

    assert_eq!(&output[..], "From closure: [1, 2, 3]\n");

    // After calling closure:
    assert_eq!(list, vec![1, 2, 3]);

    //drop(buffer);
  }
  */

  #[test]
  fn closure_can_capture_immutable_reference_for_return()
  {
    let list = vec![1, 2, 3];
    // Before defining closure:
    assert_eq!(list, vec![1, 2, 3]);

    let only_borrows = || { &list };

    // Before calling closure:
    assert_eq!(list, vec![1, 2, 3]);

    let result = only_borrows();    

    assert_eq!(*result, vec![1, 2, 3]);

    // After calling closure:
    assert_eq!(list, vec![1, 2, 3]);
  }

  #[test]
  fn closure_can_capture_mutable_reference()
  {
    let mut list = vec![1, 2, 3];

    // Before defining closure:
    assert_eq!(list, vec![1, 2, 3]);

    let mut borrows_mutably = || list.push(7);

    // We don't use the closure again after closure is called, so mutable borrow ends.
    // Between closure definition and closure call, an immutable borrow isn't allowed because no
    // other borrow are allowed when there's a mutable borrow.
    borrows_mutably();

    // After calling closure:
    assert_eq!(list, vec![1, 2, 3, 7]);
  }

  //-----------------------------------------------------------------------------------------------
  // \brief Moving Captured Values Out of Closures and the Fn Traits
  // \url https://doc.rust-lang.org/book/ch13-01-closures.html#moving-captured-values-out-of-closures-and-the-fn-traits    
  // \details 2. FnMut applies to closures that don't move captured values out of their body, but
  // that might mutate the captured values. These closures can be called more than once.
  //-----------------------------------------------------------------------------------------------

  #[test]
  fn closure_using_FnMut()
  {
    let mut list = [
      Rectangle { width: 10, height: 1 },
      Rectangle { width: 3, height: 5 },
      Rectangle { width: 7, height: 7},
    ];

    assert_eq!(list[0].width, 10);
    assert_eq!(list[1].width, 3);
    assert_eq!(list[2].width, 7);

    //---------------------------------------------------------------------------------------------
    /// sort_by_key defined to take FnMut closure because it calls the closure multiple times: once
    /// for each item in the slice.
    //---------------------------------------------------------------------------------------------

    list.sort_by_key(|r| r.width);

    assert_eq!(list[0].width, 3);
    assert_eq!(list[1].width, 7);
    assert_eq!(list[2].width, 10);
  }

  #[test]
  fn closure_using_fn_mut_that_does_not_move_value_out_of_closure()
  {
    let mut list = [
      Rectangle { width: 10, height: 1 },
      Rectangle { width: 3, height: 5 },
      Rectangle { width: 7, height: 7},
    ];

    let mut num_sort_operations = 0;
    list.sort_by_key(|r| {
      num_sort_operations += 1;
      r.width
    });

    assert_eq!(list[0].width, 3);
    assert_eq!(list[1].width, 7);
    assert_eq!(list[2].width, 10);
    assert_eq!(num_sort_operations, 6);
  }
}