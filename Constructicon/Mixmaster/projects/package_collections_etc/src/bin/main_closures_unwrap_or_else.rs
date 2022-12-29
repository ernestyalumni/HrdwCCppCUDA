//-------------------------------------------------------------------------------------------------
// cf. https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html
// Alternatives to Using match with Result<T, E>
//
// cf. https://doc.rust-lang.org/book/ch13-01-closures.html
//-------------------------------------------------------------------------------------------------

use std::fs::File;
use std::io::ErrorKind;
use std::thread;

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

  //-----------------------------------------------------------------------------------------------
  // If you want to force closure to take ownership of values it uses in the environment even
  // though the body of the closure doesn't strictly need ownership, you can use move keyword
  // before parameter list.
  //
  // This technique is mostly useful when passing a closure to a new thread to move the data so
  // that it's owned by the new thread.
  //-----------------------------------------------------------------------------------------------

  let list = vec![1, 2, 3];
  println!("Before defining closure: {:?}", list);

  // Compiler requires list be moved into closure given new thread so reference will be valid.
  thread::spawn(move || println!("From thread: {:?}", list)).join().unwrap();
}