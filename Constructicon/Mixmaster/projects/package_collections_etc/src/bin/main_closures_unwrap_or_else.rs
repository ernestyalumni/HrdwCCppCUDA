//-------------------------------------------------------------------------------------------------
// cf. https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html
// Alternatives to Using match with Result<T, E>
//-------------------------------------------------------------------------------------------------

use std::fs::File;
use std::io::ErrorKind;

fn main()
{
  // cf. https://stackoverflow.com/questions/73023600/using-unwrap-or-else-for-error-handling-in-rust
  // unwrap_or_else is for extracting Result values.
  // cf. https://doc.rust-lang.org/std/result/enum.Result.html
  // pub fn unwrap_or_else<F>(self, op: F) -> T where F: FnOnce(E) -> T
  // Returns the contained `Ok` value or computes it from a closure.
  let _greeting_file = File::open("hello.txt").unwrap_or_else(
    |error|
    {
      if error.kind() == ErrorKind::NotFound
      {
        File::create("hello.txt").unwrap_or_else(
          |error|
          {
            panic!("Problem creating the file: {:?}", error);
          })
      }
      else
      {
        panic!("Problem opening the file: {:?}", error);  
      }
    });
}