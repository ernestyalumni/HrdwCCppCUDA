//-------------------------------------------------------------------------------------------------
/// cf. https://doc.rust-lang.org/book/ch12-01-accepting-command-line-arguments.html
/// The goal is to be able to run our program with cargo run, 2 hyphens to indicate the following
/// arguments are for our program rather than for `cargo`:
/// cargo run -- searchstring example-filename.txt
//-------------------------------------------------------------------------------------------------

use std::env;

fn main() {
  let args: Vec<String> = env::args().collect();

  let query = &args[1];
  let file_path = &args[2];

  println!("Searching for {}", query);
  println!("In file {}", file_path);

  dbg!(args);    
}
