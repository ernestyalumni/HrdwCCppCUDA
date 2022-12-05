//-------------------------------------------------------------------------------------------------
// cf. https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html
// Recoverable Errors with Result
// Run as follows:
// package_collections_etc]$ cargo run --bin main_result
//-------------------------------------------------------------------------------------------------

use std::env;
use std::fs::File;

fn main()
{
  let current_working_directory = env::current_dir();
  println!("current working directory: {:#?}", current_working_directory);  

  if let Ok(current_working_directory_value) = current_working_directory
  {
    // Expected: HrdwCCppCUDA/Constructicon/Mixmaster/projects/package_collections_etc
    println!("current working directory value: {:?}", current_working_directory_value);
  }

  let greeting_file_result = File::open("hello.txt");

  // Note that, like the Option enum, the Result enum and its variants have been brought into scope
  // by the prelude, so we don't need to specify Result:: before the Ok and Err variants in the
  // match arms.

  let _greeting_file = match greeting_file_result
  {
    Ok(file) => file,
    // We want to take different actions for different failure reasons.
    // The type of value that File::open returns inside the Err variant is io::Error, which is a
    // struct provided by the standard library. This struct has a method kind that we can call to
    // get an io::ErrorKind value. The enum io::ErrorKind is provided by the standard library and
    // has variants representing the different kinds of errors that might result from an io
    // operation.
    Err(error) => match error.kind()
    {
      std::io::ErrorKind::NotFound => match File::create("hello.txt")
      {
        Ok(fc) => fc,
        Err(e) => panic!("Problem creating the file: {:?}", e),
      },
      other_error =>
      {
        panic!("Problem opening the file: {:?}", other_error);
      }
    },
  };
}