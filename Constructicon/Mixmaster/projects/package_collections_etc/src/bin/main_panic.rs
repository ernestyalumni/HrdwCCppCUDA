// cf. https://doc.rust-lang.org/book/ch09-01-unrecoverable-errors-with-panic.html
// Unrecoverable errors and panic.

// Using a panic! Backtrace
// run RUST_BACKTRACE=1 cargo run --bin main_panic
// The key to reading backtrace is to start from the top and read until you see files you wrote.
// That's the spot where the problem originated. The lines above that spot are code that your code
// has called; the lines below are code that called your code.

fn main()
{
  let v = vec![1, 2, 3];

  v[99];
}