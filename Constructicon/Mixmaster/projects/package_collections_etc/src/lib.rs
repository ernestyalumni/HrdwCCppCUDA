// https://doc.rust-lang.org/book/ch08-01-vectors.html#using-an-enum-to-store-multiple-types
// Define an enum whose variants will hold different value types, and all enum variants will be
// considered the same type: that of the enum. Then we can create a vector to hold that enum.
pub enum SpreadsheetCell
{
  Int(i32),
  Float(f64),
  Text(String),
}

//-------------------------------------------------------------------------------------------------
// \url https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html
// Propagating Errors
//-------------------------------------------------------------------------------------------------

pub mod generics_traits_lifetimes;

pub fn add(left: usize, right: usize) -> usize {
  left + right
}

pub fn add_two(a: i32) -> i32
{
  a + 2
}

#[derive(Debug)]
struct Rectangle
{
  width_: u32,
  height_: u32,
}

impl Rectangle
{
  fn can_hold(&self, other: &Rectangle) -> bool
  {
    self.width_ > other.width_ && self.height_ > other.height_
  }
}

pub fn greeting(name: &str) -> String
{
  format!("Hello {}!", name)
}

pub struct Guess
{
  value_: i32,
}

impl Guess
{
  pub fn new(value: i32) -> Guess
  {
    if value < 1 || value > 100
    {
      panic!("Guess value must be between 1 and 100, got {}.", value);
    }

    Guess { value_: value }
  }

  pub fn precise_new(value: i32) -> Guess
  {
    if value < 1
    {
      panic!("Guess value must be greater than or equal to 1, got {}.", value);
    }
    else if value > 100
    {
      panic!("Guess value must be less than or equal to 100, got {}.", value);
    }

    Guess { value_: value }
  }
}

//-------------------------------------------------------------------------------------------------
/// \brief The Anatomy of a Test Function
/// \url https://doc.rust-lang.org/book/ch11-01-writing-tests.html#the-anatomy-of-a-test-function
/// \details A test in Rust is a function that's annotated with test attribute. Attributes are
/// metadata about pieces of Rust code. To change a function into a test function, add #[test] on
/// the line before fn.
//-------------------------------------------------------------------------------------------------
#[cfg(test)]
mod tests {
  // The tests module is a regular module that follows the usualy visibility rules in Ch. 7,
  // "Paths for Referring to an Item in the Module Tree" of Rust book.
  // Because tests module is an ineer module, we need to bring the code under test in the outer
  // module into scope of inner module.
  use super::*;

  #[test]
  fn it_works() {
    let result = add(2, 2);
    assert_eq!(result, 4);
  }

  #[test]
  fn larger_can_hold_smaller()
  {
    let larger = Rectangle {
      width_: 8,
      height_: 7,
    };

    let smaller = Rectangle {
      width_: 5,
      height_: 1,
    };

    assert!(larger.can_hold(&smaller));
  }

  #[test]
  fn smaller_cannot_hold_larger()
  {
    let larger = Rectangle {
      width_: 8,
      height_: 7,
    };

    let smaller = Rectangle {
      width_: 5,
      height_: 1,
    };

    assert!(!smaller.can_hold(&larger));
  }

  #[test]
  fn it_adds_two()
  {
    assert_eq!(add_two(2), 4);
  }

  #[test]
  fn greeting_contains_name()
  {
    let result = greeting("Carol");
    // Added custom failure message composed of format string with placeholder filled in with the
    // actual value we got from the greeting function.
    assert!(
      result.contains("Carol"),
      "Greeting did not contain name, value was `{}`",
      result);
  }

  #[test]
  // Add attribute should_panic to test function. Test passes if code inside function panics.
  #[should_panic]
  fn greater_than_100()
  {
    Guess::new(200);
  }

  #[test]
  // Make should_panic test more precise, add optional expected parameter to should_panic
  // attribute; test harness will make sure failure message contains provided text.
  #[should_panic(expected = "less than or equal to 100")]
  fn less_than_100()
  {
    Guess::precise_new(200);
  }

  //-----------------------------------------------------------------------------------------------
  /// \brief Using Result<T, E> in Tests
  /// \url https://doc.rust-lang.org/book/ch11-01-writing-tests.html#using-resultt-e-in-tests
  /// \details We can write tests that use Result<T, E>. Return an Err instead of panicking.
  //-----------------------------------------------------------------------------------------------
  #[test]
  // it_works function now has the Result<(), String> return type.
  fn it_works() -> Result<(), String>
  {
    // In body of function, rather than calling assert_eq! macro, we return Ok(()) when test passes
    // and Err with String when test fails.
    if 2 + 2 == 4
    {
      Ok(())
    }
    else
    {
      Err(String::from("two plus two does not equal four"))
    }
  }
}